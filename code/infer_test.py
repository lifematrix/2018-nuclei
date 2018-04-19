
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
import skimage.morphology

def infer_part(net, image_part):
    original_shape = image_part.shape
    offset = 296 - image_part.shape[0]
    image_part = image_part/255.0
    image_part = np.reshape(image_part, (1,)+image_part.shape)
    image_part_pad = padding_array(image_part, offset, default_val=0)
    image_part_pad = np.reshape(image_part_pad, image_part_pad.shape+(1,))
    logging.info("original_shape: %s, pad.shape: %s",
                 original_shape, image_part_pad.shape)
    prob = net.infer(image_part_pad)
    prob = prob[0,...,1]
    # pred = pred > 0.5
    crop_prob = crop_to_shape(prob, original_shape)
    logging.info("original_shape: %s, pad.shape: %s, pred.shape: %s, crop_pred.shape: %s",
        original_shape, image_part_pad.shape, prob.shape, crop_prob.shape)

    return crop_prob

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

#RLE encoding for submission
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def apply_morphology(mask):
    mask = (mask*255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = (mask == 255)
    return mask

def prob_to_rles(x, cutoff=0.5):
    mask = x > cutoff
    mask = apply_morphology(mask)
    lab_img = skimage.morphology.label(mask)
    encodes = []
    for i in range(1, lab_img.max() + 1):
        # yield rle_encoding(lab_img == i)
        encodes.append(rle_encoding(lab_img == i))
    return encodes


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
    pkl_fname = "data/preprocess/stage1_2_test_set.pkl"
    with open(pkl_fname, "rb") as f:
        ds = pickle.load(f)

    net = unet.Unet(channels=1,
                    n_class=2,
                    cost='cross_entropy',
                    layers=LAYERS,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )
    net.load_weight("log/20180416/model.cpkt")

    images_code = {}
    for i, (image_id, value) in enumerate(ds.items()):
        # image_part = value[0]['img']
        # prob = infer_part(net, image_part)
        #
        # mask = prob > 0.5
        # mask_ex = apply_morphology(mask)
        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
        # ax[0].imshow(image_part, aspect="auto")
        # ax[1].imshow(mask_ex, aspect="auto")
        # ax[2].imshow(mask, aspect="auto")
        # ax[0].set_title("Input")
        # ax[1].set_title("Ground truth")
        # ax[2].set_title("Prediction")
        # fig.tight_layout()
        # plt.show()
        # if i > 3:
        #     break

        whole_image, whole_prob = infer_image(net, image_id, value)
        code = prob_to_rles(whole_prob)
        logging.info("code: %s", code)
        images_code[image_id] = code
        # if i > 5:
        #     break

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
        images_code = sorted(images_code.items(), key=lambda x:x[0])
        for image_id, codes in images_code:
            # value = np.array(value, dtype=np.int).flatten()
            for code in codes:
                f.write("%s,%s\n" % (image_id, " ".join(["%d" % x for x in code])))

def test_write_sumb():
    whole_pred = np.array([[0,1,1,0],
                           [1,0,1,1],
                           [1,0,1,1],
                           [1,0,1,0]], dtype=np.bool)
    code = encode_pred(whole_pred)
    logging.info("code: %s", code)
    images_code = {}
    images_code['abcdf'] = code
    write_subm("data/subm_20180415_01.csv", images_code)



def test_encode():
    whole_pred = np.array([[0,1,1,0],
                           [1,0,1,1],
                           [1,0,1,1],
                           [1,0,1,0]], dtype=np.bool)
    code = encode_pred(whole_pred)
    logging.info("code: %s", code)


if __name__ == "__main__":
    initlog()
    # test_write_sumb()
    images_code = infer_test()
    write_subm("data/subm_20180416_01.csv", images_code)