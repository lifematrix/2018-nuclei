
from mlog import initlog
import glob
import os
import numpy as np
import logging
import cv2
import math
import cPickle as pickle
import matplotlib.pylab as plt
from train import padding_array
from eval import crop_to_shape

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util
from train import get_dataset, unet_size, padding_array

def infer_part(net, image_part):
    original_shape = image_part.shape
    offset = 296 - image_part.shape[0]
    image_part = image_part/255.0
    image_part = np.reshape(image_part, (1,)+image_part.shape)
    image_part_pad = padding_array(image_part, offset, default_val=0)
    image_part_pad = np.reshape(image_part_pad, image_part_pad.shape+(1,))
    logging.info("original_shape: %s, pad.shape: %s",
                 original_shape, image_part_pad.shape)
    pred = net.infer(image_part_pad)
    pred = pred[0,...,1]
    pred = pred > 0.5
    crop_pred = crop_to_shape(pred, original_shape)
    logging.info("original_shape: %s, pad.shape: %s, pred.shape: %s, crop_pred.shape: %s",
        original_shape, image_part_pad.shape, pred.shape, crop_pred.shape)

    return crop_pred

def concat_parts(parts, poses, raw_shape):
    logging.info("poses: %s, raw_shape: %s", poses, raw_shape)
    whole = np.zeros(raw_shape, dtype=parts[0].dtype)

    for pos, part in zip(poses, parts):
        whole[pos[0]:pos[0]+part.shape[0], pos[1]:pos[1]+part.shape[1]] = part

    return whole


def infer_image(net, image_id, value):
    parts_pred = [ infer_part(net, x['img']) for x in value]
    whole_image = concat_parts([x['img'] for x in value], 
                               [x['pos'] for x in value],
                               value[0]['raw_shape'])
    whole_pred = concat_parts(parts_pred, 
                               [x['pos'] for x in value],
                               value[0]['raw_shape'])

    return whole_image, whole_pred

def encode_pred(pred):
    pred = np.transpose(pred).flatten()
    code = []
    flag = 0
    p = [0, 0]
    for i, x in enumerate(pred):
        if flag == 0 and x == 0:
            continue
        elif flag == 0 and x == 1:
            p[0] = i+1
            p[1] = 1 
            flag = 1
        elif flag == 1 and x == 1:
            p[1] += 1
        elif flag == 1 and x == 0:
            code.append(p)
            flag = 0
            p = [0, 0]
        else:
            pass
    if flag == 1:
        code.append(p)

    return code




def infer_test():
    LAYERS = 3
    pkl_fname = "data/preprocess/stage1_test_set.pkl"
    with open(pkl_fname, "rb") as f:
        ds = pickle.load(f)

    net = unet.Unet(channels=1,
                    n_class=2,
                    cost='cross_entropy',
                    layers=LAYERS,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )
    net.load_weight("log/20180414/model.cpkt")

    images_code = {}
    for i, (image_id, value) in enumerate(ds.items()):
        # image_part = value[0]['img']
        # mask = infer_part(net, image_part)

        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
        # ax[0].imshow(image_part, aspect="auto")
        # ax[2].imshow(mask, aspect="auto")
        # ax[0].set_title("Input")
        # ax[1].set_title("Ground truth")
        # ax[2].set_title("Prediction")
        # fig.tight_layout()
        # plt.show()

        whole_image, whole_pred = infer_image(net, image_id, value)
        code = encode_pred(whole_pred)
        logging.info("code: %s", code)
        images_code[image_id] = code
        if i > 10:
            break
        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
        # ax[0].imshow(whole_image, aspect="auto")
        # ax[2].imshow(whole_pred, aspect="auto")
        # ax[0].set_title("Input")
        # ax[1].set_title("Ground truth")
        # ax[2].set_title("Prediction")
        # fig.tight_layout()
        # plt.show()


    return images_code

def write_subm(fname, images_code):
    with open(fname, "w") as f:
        f.write("ImageId,EncodedPixels\n")
        for image_id, value in images_code:
            f.write("%s,%s", image_id, " ".join([" ".join(x) for x in value]))


def test_encode():
    whole_pred = np.array([[0,1,1,0],
                           [1,0,1,1],
                           [1,0,1,1],
                           [1,0,1,0]], dtype=np.bool)
    code = encode_pred(whole_pred)
    logging.info("code: %s", code)


if __name__ == "__main__":
    initlog()
    # test_encode()
    images_code = infer_test()
    write_subm("data/subm_20180415_01.csv", images_code)