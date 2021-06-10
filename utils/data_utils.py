import os
from io import BytesIO
from tempfile import TemporaryDirectory

import tensorflow as tf
from PIL import Image
from google.cloud import storage
import random

class GCSTFRecordWriter(object):
    def __init__(self, fn, buffer_size=1, auto_close=False):
        """
        Shuffle things in the shuffle buffer and write to tfrecords

        If buffer_size == 0 then no shuffling
        :param fn:
        :param buffer_size:
        """
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.python_io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.python_io.TFRecordWriter(fn)
        self.buffer_size = buffer_size
        self.buffer = []
        self.auto_close=auto_close

    def write(self, x):
        if self.buffer_size < 10:
            self.writer.write(x)
            return

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(x)
        else:
            random.shuffle(self.buffer)
            for i in range(self.buffer_size // 5):  # Pop 20%
                self.writer.write(self.buffer.pop())

    def close(self):
        # Flush buffer
        if self.buffer_size > 1:
            random.shuffle(self.buffer)
        for x in self.buffer:
            self.writer.write(x)

        self.writer.close()
        if self.gclient is not None:
            print("UPLOADING!!!!!", flush=True)
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        if self.auto_close:
            print("CALLING CLOSE")
            self.close()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def pil_image_to_jpgstring(image: Image):
    """
    :param image: PIL image
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=95)
        return output.getvalue()

def get_size_for_resize(image_size, shorter_size_trg=384, longer_size_max=512):
    """
    Gets a new size for the image. Try to do shorter_side == shorter_size_trg. But we make it smaller if the longer
    side is > longer_size_max.
    :param image_size:
    :param shorter_size_trg:
    :param longer_size_max:
    :return:
    """

    w, h = image_size
    size = shorter_size_trg  # Try [size, size]

    if min(w, h) <= size:
        return w, h

    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * size > longer_size_max:
        size = int(round(longer_size_max * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return w, h
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return ow, oh