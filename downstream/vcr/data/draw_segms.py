import gc
import json
import argparse
from tqdm import tqdm
from downstream.vcr.data.colormap import color_list
from PIL import Image
import PIL.ImageDraw as ImageDraw

TRANSPARENCY = .15
OPACITY = int(255 * TRANSPARENCY)


parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds (corresponding to both the number of training files and the number of testing files)',
)
parser.add_argument(
    '-split',
    dest='split',
    default='train',
    type=str,
)
parser.add_argument(
    '-mode',
    dest='mode',
    default='answer',
    type=str,
)
args = parser.parse_args()


split = args.split
mode = args.mode
save_dir = f'segm/{split}/{mode}'

VCR_DIRECTORY = ''
items = [json.loads(s) for s in open(f'{VCR_DIRECTORY}/annotation/{split}.jsonl', 'r')]
img_dir = f'{VCR_DIRECTORY}/vcr1images'

counter = 0
for i, item in enumerate(tqdm(items)):
    if i % args.num_folds != args.fold:
        continue
    counter += 1

    mentions = []
    objects = []

    for word in item["question"]:
        if isinstance(word, list):
            mentions.extend([w for w in word if item["objects"][w] == "person"])
            objects.extend([w for w in word if item["objects"][w] != "person"])

    for ans in item["answer_choices"]:
        for word in ans:
            if isinstance(word, list):
                mentions.extend([w for w in word if item["objects"][w] == "person"])
                objects.extend([w for w in word if item["objects"][w] != "person"])

    if mode == 'rationale':
        for rat in item["rationale_choices"]:
            for word in rat:
                if isinstance(word, list):
                    mentions.extend([w for w in word if item["objects"][w] == "person"])
                    objects.extend([w for w in word if item["objects"][w] != "person"])

    mentions = list(set(mentions))
    objects = list(set(objects))

    image = Image.open(f'{img_dir}/{item["img_fn"]}').convert("RGBA")
    meta = json.load(open(f'{img_dir}/{item["metadata_fn"]}', 'r'))
    segms = meta['segms']

    for i, segm in enumerate(segms):
        if i in mentions:
            color = color_list[:-1][i % (len(color_list) - 1)]
        elif i in objects:
            color = color_list[-1]
        else:
            continue

        overlay = Image.new('RGBA', image.size, tuple(color) + (0,))
        draw = ImageDraw.Draw(overlay)
        for segm_part in segm:
            if len(segm_part) < 2:
                segm_part += tuple([segm_part[0]])

            segm_part = tuple(tuple(x) for x in segm_part)
            draw.polygon(segm_part, fill=tuple(color) + (OPACITY,))

        draw = ImageDraw.Draw(image)
        for segm_part in segm:
            segm_part = tuple(tuple(x) for x in segm_part)
            segm_part += tuple([segm_part[0]])
            draw.line(segm_part, fill=tuple(color), width=7)
        image = Image.alpha_composite(image, overlay)

    image = image.convert("RGB")
    image.save(f'{save_dir}/{item["annot_id"]}.jpg')

    gc.collect()

print(f'writing {counter} examples')

