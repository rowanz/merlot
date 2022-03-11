"""Serve predictions from Grover."""
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-size', type=str, default="medium")
parser.add_argument('-tag', type=str, default="")
parser.add_argument('-batch_size', type=int, default=1)

args = parser.parse_args()
GPUID = args.gpu
SIZE = args.size
TAG = "-" + args.tag if args.tag else ''

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import flask
from flask_cors import CORS
import tensorflow as tf
import sys

from lm.modeling import GroverConfig, sample_seq2seq
from sample.encoder import get_encoder, extract_generated_target
import logging
from datetime import datetime
import click
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd


app = flask.Flask(__name__, template_folder='.')
CORS(app, resources={r'/api/*': {'origins': '*'}})

logger = logging.getLogger(__name__)

# SETUP
encoder = get_encoder()
news_config = GroverConfig.from_json_file(f'lm/configs/{SIZE}.json')
batch_size = args.batch_size
top_p = 0.94

# gsutil cp gs://merlot/denoisify_ckpt/model=medium~lr=1e-5~steps=80000~v0/model.ckpt-80000\* ckpt-medium/
# cd ckpt-medium
# mv model.ckpt-80000.index model.ckpt.index
# mv model.ckpt-80000.data-00000-of-00001 model.ckpt.data-00000-of-00001
# mv model.ckpt-80000.meta model.ckpt.meta

def _prepare_instance(instance, target='cleanasr'):
    """
    Process each instance
    :param instance:
    :param target:
    :return:
    """

    context_formatted = [encoder.begin_title] + encoder.encode(instance['noisyasr'])
    if target == 'noisyasr':
        instance['eos_token'] = encoder.end_title
    else:
        # target == cleanasr
        context_formatted += [encoder.end_title, encoder.begin_article]
        instance['eos_token'] = encoder.end_article

    if len(context_formatted) > 1280:
        context_formatted = context_formatted[-1280:]
    print("CTX is {}".format(encoder.decode(context_formatted)), flush=True)
    instance['context_formatted'] = context_formatted
    return instance


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=tf.Graph()) as sess:
    initial_context = tf.placeholder(tf.int32, [batch_size, None])
    eos_token = tf.placeholder(tf.int32, [])
    ignore_ids = tf.placeholder(tf.bool, [news_config.vocab_size])

    tokens, probs = sample_seq2seq(news_config=news_config, initial_context=initial_context,
                           eos_token=eos_token, ignore_ids=ignore_ids, p_for_topp=top_p, max_len=1537)

    saver = tf.train.Saver()
    if not os.path.exists(f'ckpt-{SIZE}{TAG}'):
        raise ValueError(f'download ckpt-{SIZE}{TAG}')
    saver.restore(sess, f'ckpt-{SIZE}{TAG}/model.ckpt')

    # create a server endpoint to answer requests
    print("READY FOR GENERATION", flush=True)


    @app.route('/', methods=['GET'])
    def form_ask():
        """Return the demo page."""
        return flask.render_template('index.html', model_details="The model used is Grover {} with a max sequence length of 1536 tokens. It was trained on 20M denoise examples for 1 epoch (80k iterations). using nucleus sampling p=0.94".format(args.size))


    @app.route('/api/ask', methods=['POST'])
    def api_ask():
        """Serve a prediction for a single instance."""
        instance = dict(flask.request.json)
        print("GOT A REQUEST for {}".format(instance), flush=True)

        target = instance.get('target', 'cleanasr')
        instance = _prepare_instance(instance, target=target)
        if instance is None:
            return flask.jsonify({
                'instance': instance,
                'gen': 'error',
            }), 200

        eos_token_val = instance.pop('eos_token')
        context_formatted = instance.pop('context_formatted')

        # Indices we definitely DONT WANT TO PREDICT
        ignore_ids_np = np.array(encoder.special_tokens_onehot)
        ignore_ids_np[eos_token_val] = 0

        tokens_out, probs_out = sess.run([tokens,probs], feed_dict={initial_context: np.stack([context_formatted]*batch_size),
                                          eos_token: eos_token_val,
                                          ignore_ids: ignore_ids_np})
        out_decoded = extract_generated_target(
            output_tokens=tokens_out[0], encoder=encoder,
            target={'noisyasr': 'title', 'cleanasr': 'article'}[target])['extraction'].strip()
        print("SENDING BACK {}".format(out_decoded), flush=True)

        ctx_ppl = float(np.exp(-np.mean(np.log(probs_out[0,:max(len(context_formatted) - 1, 1)]))))

        new_instance = {k: v for k, v in instance.items()}
        new_instance[target] = out_decoded
        new_instance['size'] = SIZE
        new_instance['tag'] = TAG.strip('-')
        new_instance['top_p'] = top_p
        new_instance['ctx_ppl'] = ctx_ppl

        with open(f'log{GPUID}.jsonl', 'a+') as logfile:
            logfile.write(json.dumps(new_instance) + '\n')

        return flask.jsonify({
            'instance': instance,
            'gen': out_decoded,
            'ppl': ctx_ppl,
        }), 200

    @click.command()
    def serve():
        """Serve predictions on port 5000."""
        logging.basicConfig(
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            level=logging.INFO)
        logger.info('Running prod server on http://127.0.0.1:5000/')


    WSGIServer(('0.0.0.0', 4999 + GPUID), app).serve_forever()
