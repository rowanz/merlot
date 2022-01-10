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
save_dir = f'bbox/{split}/{mode}'

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
    boxes = meta['boxes']

    for i, box in enumerate(boxes):
        if i in mentions:
            color = color_list[:-1][i % (len(color_list) - 1)]
        elif i in objects:
            color = color_list[-1]
        else:
            continue

        box = [int(x) for x in box[:4]]
        x1, y1, x2, y2 = box
        shape = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

        overlay = Image.new('RGBA', image.size, tuple(color) + (0,))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(shape, fill=tuple(color) + (OPACITY,))

        draw = ImageDraw.Draw(image)
        draw.line(shape, fill=tuple(color), width=7)

        image = Image.alpha_composite(image, overlay)

    image = image.convert("RGB")
    image.save(f'{save_dir}/{item["annot_id"]}.jpg')

    gc.collect()

print(f'writing {counter} examples')

