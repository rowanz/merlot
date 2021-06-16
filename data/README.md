# MERLOT Data for pretraining (YT-Temporal-180M)

* `process.py` contains a quick script for turning the `example_video` into a very small tfrecord for pretraining.
* The dataset is available for academic use, please contact Rowan for access. We probably cannot release the videos (for legal reasons and to protect privacy). What we are releasing are annotations that look like this

* `denoised`: a list of spans of `noisyasr` text, that was cleaned up with a finetuned Grover model (output is `cleanasr`). The perplexity of the context is under `ctx_ppl`
* `info`: a dictionary of info with information about the YouTube video
* `subtitles`: Each word, along with the approximate timestamp about when it was said in the video
* `_te`: Time elapsed (this isn't needed at all)
