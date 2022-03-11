# MERLOT Data for pretraining (YT-Temporal-180M)

The dataset is available for academic use, please see [https://rowanzellers.com/merlot](https://rowanzellers.com/merlot) for access information. We do not plan to release the videos, just their metadata (like the video IDs, titles, and descriptions).

## Preprocessing data
`process.py` contains a quick script for turning the `example_video` into a very small tfrecord for pretraining. You'll need to download videos into a similar format (including the Grover denoisified transcript), or change the script, to get things to work.

## Grover denoisification
We've uploaded denoised transcripts as part of the dataset release. If you'd like the denoiser model we used, see the [data/groverdenoise](data/groverdenoise) folder. Our checkpoint is publicly available at `gs://merlot/denoisify_ckpt/model=medium~lr=1e-5~steps=80000~v0/model.ckpt-80000\*` -- you'll need to download that to your computer, and run [data/groverdenoise/run_server.py](data/groverdenoise/run_server.py), which should let you interact with the model.
