"""
Go from video files to tfrecord files.

We will load from the "example_video" folder and create a dataset of just 1 video for pretraining
You'll need to change this script for more videos
"""

import sys

sys.path.append('../')

import argparse
import csv
import tempfile
import hashlib
import json
import numpy as np
from tqdm import tqdm
import time
from data.video_utils import extract_all_frames_from_video, extract_frames_from_video
from utils.encode.encoder import get_encoder
from utils.data_utils import *
import string
import editdistance
import tslearn.metrics
import pandas as pd
import shutil
import atexit
import ftfy
import regex as re
import demoji

encoder = get_encoder()

CHUNK_LEN = 31
STOP_THRESH = 0.75  # if we are >stop_thresh through a chunk already and we see a stopword, exit now
NUM_CHUNKS = 16

info_fn = 'example_video/WAaKRUoY6Io.grover.json'
video_fn = 'example_video/WAaKRUoY6Io.mp4'

#######################################################

with open(info_fn, 'r') as f:
    video = json.load(f)

TRANSLATOR = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def align_using_dtw(input_asr, grover_output, radius_perc=0.1, radius_abs=32):
    """
    :param input_asr: List of words
    :param grover_output: List of words also, could be different size
    :param radius_perc: Percent of input ASR
    :param radius_abs: Absolute ntokens
    :return:
    """
    max_radius = int(max(len(input_asr) * radius_perc, radius_abs))
    # sometimes grover just keeps going
    if len(grover_output) > len(input_asr):
        grover_output = grover_output[:len(input_asr) + max_radius]

    # DONT give the alignment freedom if it's at the end of a sequence to just "give up" by padding with zeros
    # Default value is high
    auto2other = np.zeros((len(input_asr), len(grover_output)), dtype=np.float32) + 9999.0

    def _preprocess_text(x):
        return x.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    input_asr_pre = [_preprocess_text(x) for x in input_asr]
    input_gro_pre = [_preprocess_text(x) for x in grover_output]
    for a_idx, a in enumerate(input_asr_pre):
        start = max(a_idx - max_radius, 0)
        end = min(a_idx + max_radius, len(input_gro_pre))
        for o_idx in range(start, end):
            o = input_gro_pre[o_idx]
            auto2other[a_idx, o_idx] = editdistance.eval(a, o)

    idxs, score = tslearn.metrics.dtw_path_from_metric(auto2other, metric='precomputed')
    denoised_out = [[] for x in input_asr]
    has_seen = -1
    for idx1, idx2 in idxs:
        if (idx1 >= len(input_asr)) or (idx2 >= len(grover_output)):
            break
        if idx2 > has_seen:
            # Basically don't add if it's a duplicate -- a grover output that matches to 2 things
            # This often leads to slightly weird results because we really should match the next thing, but we instead matched the first thing
            # e.g.
            # input_asr_pre = ['much', 'of', 'a', 'pancake', 'waffle', 'person', 'so', 'i', 'love', 'a']
            # input_gro_pre = ['much', 'of', 'a', 'pancakewaffle', 'person', 'so', 'i', 'love', 'a', 'good']
            # but we align pancakewaffle-> pancake and person -> waffle AND person -> person
            denoised_out[idx1].append(grover_output[idx2])
        has_seen = idx2
    return [' '.join(x) for x in denoised_out]

def clean_subtitles(subtitle_dicts):
    """
    :param subtitle_dicts: {'word': X, 'time': Y}
    :return:
    """
    # Remove &gt;&gt; maybe using ftfy or something
    new_dicts = []
    for x in subtitle_dicts:
        if x['word'].startswith('&') or x['word'].endswith(';'):
            continue
        fixed_word = ftfy.ftfy(x['word'])
        if len(fixed_word) == 0:
            continue
        x['word'] = fixed_word
        new_dicts.append(x)
    return new_dicts

def clean_description(text):
    # Strip emojis first
    all_emojis = demoji.findall(text)
    for k, v in all_emojis.items():
        text = text.replace(k, f'[{v}]'.replace(' ', ''))
    text = text.strip()

    # Remove URLs
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/11332580
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "%", text)

    text = re.sub(' +', ' ', text) # Probably should have done
    text = re.sub('\s*\n+', '\n', text)
    text = text.strip()
    return text


def split_video_into_chunks(item):
    """
    :param item
    :return:
    """
    item['subtitles'] = clean_subtitles(item['subtitles'])
    vtt = pd.DataFrame(item['subtitles'])
    if 'word' not in vtt.columns:
        raise ValueError(f"'Word' not in item['subtitles'] \n{item}")
    if 'title' not in item['info']:
        raise ValueError(f"'title' not in item['info'] \n{item}")

    vtt['encoded'] = vtt['word'].apply(lambda x: encoder.encode(' ' + x.strip()))

    # Any punctuation gets attached to a nearby word by design here
    denoised_word_by_word = []
    for x in item['denoised']:
        # Ftfy just in case
        cleanasr = ftfy.ftfy(x['cleanasr'])
        denoised_word_by_word += cleanasr.split(' ')

    # Align
    vtt['denoised'] = align_using_dtw(vtt['word'], denoised_word_by_word)
    vtt['denoised_encoded'] = [encoder.encode(f' {x}') if len(x) > 0 else [] for x in vtt['denoised'].tolist()]

    chunks = []
    start_idx = 0
    clean_enc_buffer = []
    noisy_enc_buffer = []

    # I don't think this is ever really needed, it's just to guard against bugs
    MAX_TS = item['info'].get('duration', 1000000) -1.0

    for idx, row in vtt.iterrows():
        clean_enc_buffer += row['denoised_encoded']
        noisy_enc_buffer += row['encoded']

        if idx < (vtt.shape[0] - 1):
            noisy_len_after = len(noisy_enc_buffer) + len(vtt.loc[idx + 1, 'encoded'])
            clean_len_after = len(clean_enc_buffer) + len(vtt.loc[idx + 1, 'denoised_encoded'])

            commit_now = max(noisy_len_after, clean_len_after) > CHUNK_LEN
            commit_now = commit_now or (row['denoised'].endswith(('.', '?', '!')) and len(clean_enc_buffer) >= (
                    CHUNK_LEN * STOP_THRESH))
        else:
            commit_now = True

        # there might be some weirdness with the end timestep but whatevs. I want to make sure nothing overlaps though
        if commit_now:
            mean_timestep = (vtt.loc[start_idx, 'time'] + vtt.loc[idx, 'time']) / 2.0
            if mean_timestep < MAX_TS:
                chunks.append({
                    'start': start_idx,
                    'end': idx,
                    'clean_enc': clean_enc_buffer,
                    'noisy_enc': noisy_enc_buffer,
                    'is_eoc': False,
                    'mean_timestep': mean_timestep,
                })
            clean_enc_buffer = []
            noisy_enc_buffer = []
            start_idx = idx + 1
    if len(chunks) == 0:
        raise ValueError("chunks is empty")
    chunks[-1]['is_eoc'] = True

    return chunks


def video_chunk_iterator():
    """
    You'd need to change this for multiple videos
    :return:
    """
    chunks = split_video_into_chunks(video)
    # Extract frames at each chunk
    frames = extract_frames_from_video(video_file=video_fn,
                                       times=[x['mean_timestep'] for x in chunks], use_multithreading=True,
                                       info=video['info'])
    trg_size = get_size_for_resize((frames.shape[2], frames.shape[1]), shorter_size_trg=384,
                                   longer_size_max=512)

    for i, frame_i in enumerate(frames):
        img_i = Image.fromarray(frame_i, mode='RGB')
        if trg_size != img_i.size:
            img_i = img_i.resize(trg_size, resample=Image.BICUBIC)

        # Put the frame as well as global stuff in there
        chunks[i]['chunk_num'] = i
        chunks[i]['frame'] = img_i
        chunks[i]['video_id'] = video['info']['id']
    yield chunks

def buffered_chunk_iterator():
    buffer = []
    for chunk in video_chunk_iterator():
        buffer.extend(chunk)
        while len(buffer) >= NUM_CHUNKS:
            yield buffer[:NUM_CHUNKS]
            buffer = buffer[NUM_CHUNKS:]

train_file = 'out.tfrecord'

num_written = 0
st = time.time()
with GCSTFRecordWriter(train_file, buffer_size=10000, auto_close=False) as train_writer:
    for chunks in buffered_chunk_iterator():
        feats = {}
        for i, c_i in enumerate(chunks):
            image_encoded = pil_image_to_jpgstring(c_i['frame'])
            current_feats = {
                'image/encoded': bytes_feature(image_encoded),
                'image/height': int64_feature(c_i['frame'].height),
                'image/width': int64_feature(c_i['frame'].width),
                'image/key/sha256': bytes_feature(hashlib.sha256(image_encoded).hexdigest().encode('utf-8')),
                'image/format': bytes_feature('jpeg'.encode('utf-8')),
                'youtube_id': bytes_feature(c_i['video_id'].encode('utf-8')),
                'tokenized_cleaned_asr': int64_list_feature(c_i['clean_enc']),
                'tokenized_raw_asr': int64_list_feature(c_i['noisy_enc']),
                'is_eoc': int64_feature(int(c_i['is_eoc'])),
                'mean_time': float_list_feature([c_i['mean_timestep']]),
                'chunk_num': int64_feature(c_i['chunk_num']),
            }
            for k, v in current_feats.items():
                feats[f'c{i:02d}/{k}'] = v

        example = tf.train.Example(features=tf.train.Features(feature=feats))
        train_writer.write(example.SerializeToString())
        num_written += 1
        if num_written % 10 == 0:
            te = time.time() - st
            print(f"Wrote {num_written} in {te:.3f}", flush=True)
    te = time.time() - st
    print(f"Wrote {num_written} in {te:.3f}", flush=True)
    train_writer.close()
