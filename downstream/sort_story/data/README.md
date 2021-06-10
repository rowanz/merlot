## What's in here?

This dataset reports results on the story ordering corpus:

https://arxiv.org/pdf/1606.07493.pdf

This repo creates tfrecords corresponding to each possible ordering of
each possible story in the validation/test sets.

http://visionandlanguage.net/

## how to run?

1. download.py to get data/images
2. python make_tfrecords.py data/sis/val.story-in-sequence.json


python make_tfrecord.py data/sis/test.story-in-sequence.json data/test/ --num_threads 64 --just_one_perm 0
python make_tfrecord.py data/sis/val.story-in-sequence.json data/images/val --num_threads 64 --just_one_perm 0

python make_tfrecord.py data/sis/test.story-in-sequence.json data/test/ --num_threads 64 --just_one_perm 1
python make_tfrecord.py data/sis/val.story-in-sequence.json data/images/val --num_threads 64 --just_one_perm 1

## there are a few corrupt jpgs, which we had to fix by converting.

within test:

convert 764378.jpg 764378_new.jpg; mv 764378_new.jpg 764378.jpg

within val:

convert 4765124.jpg 4765124_new.jpg ; mv 4765124_new.jpg 4765124.jpg
convert 366831957.gif 366831957.jpg ; mv 366831957.gif backup366831957.gif
convert 764437.jpg 764437_new.jpg ; mv 764437_new.jpg 764437.jpg