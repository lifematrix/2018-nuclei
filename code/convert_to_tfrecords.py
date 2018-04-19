"""
Convert a image & mask file list to tf records
"""

import tensorflow as tf
import argparse
import sys
import os
import math

from aivis.base.fileutil import read_flist
from aivis.data.ImageReader import ImageReader
from train import get_dataset, unet_size, padding_array
import logging
from mlog import initlog
import cv2
import numpy as np


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
    values: A scalar or list of values.

    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
    values: A string.

    Returns:
    a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_mask_to_tfexample(image_data, image_format, height, width, mask_data, mask_format, image_id=""):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/file': bytes_feature(image_id),
      'mask/encoded': bytes_feature(mask_data),
      'mask/format': bytes_feature(mask_format),
    }))

#
# def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, n_shards):
#     output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
#       tfrecord_filename, split_name, shard_id, n_shards)
#     # return os.path.join(dataset_dir, output_filename)
#     return output_filename
#

def convert_dataset(images, masks, tfrecord_filename, n_shards):
    """Converts the given filenames to a TFRecord dataset.

    Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
    (integers).
    dataset_dir: The directory where the converted datasets are stored.
    """
    num_per_shard = int(math.ceil(len(images) / float(n_shards)))
    logging.info("num_per_shard: %s, len(images): %s, n_shards: %s", num_per_shard, len(images), n_shards)

    with tf.Graph().as_default() as graph:
        image_reader = ImageReader()

        with tf.Session(graph=graph) as sess:

            for shard_id in range(n_shards):
                output_filename = "%s_%05d-of-%05d.tfrecord" % (tfrecord_filename, shard_id+1, n_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(images))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(images), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = cv2.imencode(".png", images[i])[1].tostring()
                        height, width = images[i].shape[:2]
                        mask_data = cv2.imencode(".png", masks[i])[1].tostring()



                        example = image_mask_to_tfexample(
                            image_data, 'png', height, width, mask_data, "png", image_id=str(i))
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def main():
    LAYERS = 3
    pkl_fname = "data/preprocess/stage1_train_set_rgb.pkl"
    images, masks = get_dataset(pkl_fname)
    logging.info("read train set: %s, %s", images.shape, masks.shape)
    logging.info("image:[%s, %s], mask:[%s, %s]", np.max(images), np.min(images), np.max(masks), np.min(masks))

    # pred_size, offset = unet_size(256, LAYERS)
    # logging.info("pred_size: %d, offset: %d", pred_size, offset)
    # images = padding_array(images, offset, default_val=0.0)
    # masks = padding_array(masks, offset, default_val=False)


    # args.data_dir = args.data_dir.strip()
    # if len(args.data_dir) >= 0:
    #     fnames = [os.path.join(args.data_dir, x) for x in fnames]

    train_ratio = 0.9
    n_train = int(len(images)*train_ratio)
    logging.info("train_ratio: %s, n_train: %s, n_val: %s", train_ratio, n_train, len(images)-n_train)
    convert_dataset(images[:n_train], masks[:n_train], "data/tfrecords/256x256/train", 4)
    convert_dataset(images[n_train:], masks[n_train:], "data/tfrecords/256x256/val", 2)

if __name__ == "__main__":
    initlog("log")
    main()
