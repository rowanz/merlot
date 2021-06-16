"""
Downloads checkpoint
Use the argument if you want 5 segments (used for sort story) otherwise we use 4 segments (we used this for everything else)
"""

import os
import requests
import argparse

parser = argparse.ArgumentParser(description='Download MERLOT!')
parser.add_argument(
    '-use_5_segments',
    dest='use_5_segments',
    action='store_true'
)
use_5_segments = parser.parse_args().use_5_segments
nseg = 5 if use_5_segments else '4'
model_dir = os.path.join('checkpoints', f'checkpoint_{nseg}segments/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for ext in ['data-00000-of-00001', 'index', 'meta', 'checkpoint']:
    r = requests.get(f'https://storage.googleapis.com/merlot/checkpoint_{nseg}segments/model.ckpt.{ext}', stream=True)
    with open(os.path.join(model_dir, f'model.ckpt.{ext}'), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = min(1000, file_size)
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    print(f"Just downloaded merlot/checkpoint_{nseg}segments/model.ckpt.{ext}!", flush=True)
