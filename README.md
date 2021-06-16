# merlot
[MERLOT: Multimodal Neural Script Knowledge Models](https://arxiv.org/abs/2106.02636)

MERLOT is a model for learning what we are calling "neural script knowledge" -- representations about what is going on in videos, spanning multiple video frames with associated captions.


Visit our project page at [rowanzellers.com/merlot](https://rowanzellers.com/merlot), or read the [full paper](https://arxiv.org/abs/2106.02636) to learn more.

![teaser](https://i.imgur.com/RD6yb9E.png "teaser")

## What's here

We are releasing the following:
* Code for the MERLOT model (in [model/](model/), with data processing in [data/](data/)
* Code for running MERLOT over visual story ordering.

We plan to release:
* Information about the videos used in this work
* Code for adapting the model to other tasks (not strictly needed, but just to make things easier)

This is somewhat ongoing -- we hope to make it somewhat easier to adapt MERLOT to other tasks, please follow if interested!

## Enviroment and setup

There are two different ways of running MERLOT right now
* **Pretraining on videos** This requires a TPU pod.
* **Finetuning on downstream tasks** We did this on TPU v3-8 machines. You can in theory do this on GPUs, however, this isn't tested or officially supported right now.
* **Zero-shot visual-story ordering** I have code for this on a TPU, but you should be able to do this on a GPU too.


```bash
conda create --name merlot python=3.7 && conda activate merlot
conda install -y python=3.7 tqdm numpy pyyaml scipy ipython cython typing h5py pandas

# If running on GPU
pip install tensorflow-gpu==1.15.5
# If running on TPU
pip install tensorflow==1.15.5

pip install --upgrade google-api-python-client oauth2client boto3 cloud-tpu-profiler regex opencv-python-headless Pillow seaborn
pip install numpy==1.17.0
```

### Pretraining from scratch
This requires a large TPU pod for data-parallelism.
* First, you'll need to get a bunch of training data in "tfrecord" format --  see data processing in [data/](data/) for that. You'll then need to adjust the configuration of [model/configs/merlot.yaml](model/configs/merlot.yaml) accordingly. You'll also need to add in your output path (where you want your newly pretrained model to be saved).
* Next, in the `model` directory, run `python train.py configs/merlot.yaml`

### Finetuning on downstream tasks
* You can download our checkpoint using [download_checkpoint.py](download_checkpoint.py). There are two options -- we used a checkpoint with 4 frame-caption segments for general purpose pretraining, and then we trained it for longer (using 5 frame-caption segments) to adapt to the story ordering task. 

  We suggest using the *4 segments* checkpoint because that's what we used for all of our finetuning experiments. This corresponds to the configuration at We used the configuration [model/merlot.yaml](model/merlot.yaml).
* Actual finetuning code TBD -- you just create a `MerlotModel` [model/modeling.py](model/modeling.py), set up your finetuning task (usually involving an additional output layer), and finetune.


### Bibtex
```
@article{zellersluhessel2021merlot,
    title={MERLOT: Multimodal Neural Script Knowledge Models},
    author={Zellers, Rowan and Lu, Ximing and Hessel, Jack and Yu, Youngjae and Park, Jae Sung and Cao, Jize and Farhadi, Ali and Choi, Yejin},
    journal={arXiv preprint arXiv:2106.02636},
    year={2021}
}
```

