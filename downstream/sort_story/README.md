## What's in here?
Zero-shot experiments on the [VIST/SIND dataset](https://visionandlanguage.net/VIST/). The format is the following:

Given 5 image-caption pairs that together tell a narrative, put the images in the right order *given the captions in the right order.* This differs from the [Sort Story paper](https://www.aclweb.org/anthology/D16-1091/) because we're assuming the captions are already in the right order. We're making this assumption because it doesn't advantage giant language models which might be able to solve the task given just the captions.

## Running these experiments

1. First, you need to go to the configuration at [../../model/configs/merlot_5segments.yaml](../../model/configs/merlot_5segments.yaml) and fill in `output_dir` -- that's where the model will load the checkpoint. 

    You can use our pretrained checkpoint at `gs://merlot/checkpoint_5segments/`. Just run `gsutil cp gs://merlot/checkpoint_5segments/* ${CHECKPOINT_PATH}`

2. Next, grab the data. You can use the [data](data) directory, or use what we processed -- it's in `tfrecord` format at `gs://merlot/data/sort_story/val/`. You'll need to copy it to your own bucket if using a TPU, since otherwise it won't be in the right zone.

3. Finally, run the script `python get_zero_shot_logits.py` to get the scores from the model for each story, and `python score_permutations.py` to score all stories (considering all 5! orderings per story there).