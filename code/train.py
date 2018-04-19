

from mlog import initlog
import glob
import os
import numpy as np
import logging
import cv2
import math
import cPickle as pickle
import matplotlib.pylab as plt
import gc

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util

def get_dataset(pkl_fname):
    with open(pkl_fname, "rb") as f:
        ds = pickle.load(f)

    n_parts = sum(len(v) for v in ds.values())
    logging.info("n_parts: %d", n_parts)

    images = []
    masks = []

    n_err = 0
    for key, value in ds.items():
        for i, d in enumerate(value):
            if np.max(d['img']) == 0:
                logging.info('skip: %s, # %s', key, i)
                n_err += 1
                continue
            images.append(d['img'])
            masks.append(d['mask'])

    images = np.array(images)
    masks = np.array(masks, dtype=np.uint8)
    # images = scipy.sparse.csr_matrix(images, dtype=np.float32)
    # masks = scipy.sparse.csr_matrix(masks, dtype=np.bool)
    # max_v = np.max(images)
    # if max_v > 0.0:
    #     images /= max_v

    del ds
    logging.info("valid images: #%d, err: #%d", len(images), n_err)
    return images, masks


def normalize(images):
    images_r = np.zeros(images.shape, dtype=np.float32)
    max_v = float(np.max(images))
    if max_v > 0.0:
        # images /= max_v
        images_r = images/max_v
    return images_r

def test_data_provider(data_provider):
    x_test, y_test = data_provider(1)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
    ax[0].imshow(x_test[0, ..., 0], aspect="auto")
    gt = y_test[0, ..., 1]
    # mask = prediction[0, ..., 1] > 0.5
    # gt_cropped = crop_to_shape(gt, mask.shape)
    ax[1].imshow(gt, aspect="auto")
    #ax[2].imshow(mask, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    #ax[2].set_title("Prediction")
    logging.info("test_data_provider:: image:[%s, %s], mask:[%s, %s]", np.max(x_test[0, ..., 0]), np.min(x_test[0, ..., 0]), np.max(gt), np.min(gt))
    fig.tight_layout()
    plt.show()


def test_data(images, masks, i):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
    ax[0].imshow(images[i, ...], aspect="auto")
    gt = masks[i, ...]
    # mask = prediction[0, ..., 1] > 0.5
    # gt_cropped = crop_to_shape(gt, mask.shape)
    ax[1].imshow(gt, aspect="auto")
    #ax[2].imshow(mask, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    #ax[2].set_title("Prediction")
    fig.tight_layout()
    plt.show()

def padding_array(ary, offset, default_val):
    logging.info("ary.shape: %s", ary.shape)
    gc.collect()
    p1 = offset/2
    p2 = offset - p1

    psize = [(p1, p2),(p1, p2)]
    if len(ary.shape) == 3:
        psize.insert(0, (0, 0))
    elif len(ary.shape) == 4:
        psize.insert(0, (0, 0))
        psize.append((0, 0))

    logging.info("psize: %s", psize)
    ary_padded = np.pad(ary, psize, "constant", constant_values=default_val)

    return ary_padded



def unet_size(s, L):
    s0 = s
    for i in range(L - 1):
        s = (s - 4) / 2
        print("layer %d, %d" % (i + 1, s))
    s -= 4
    print("Layer %d, %d" % (L, s))
    for i in range(L - 2, -1, -1):
        s = s * 2 - 4
        print("Layer %d, %d" % (i + 1, s))

    return s, s0 - s

def main():
    np.random.seed(12345)
    LAYERS = 3
    pkl_fname = "data/preprocess/stage1_train_set_rgb.pkl"
    images, masks = get_dataset(pkl_fname)
    logging.info("read train set: %s, %s", images.shape, masks.shape)
    logging.info("image:[%s, %s], mask:[%s, %s]", np.max(images), np.min(images), np.max(masks), np.min(masks))

    pred_size, offset = unet_size(256, LAYERS)
    logging.info("pred_size: %d, offset: %d", pred_size, offset)
    images = padding_array(images, offset, default_val=0.0)
    masks = padding_array(masks, offset, default_val=False)
    logging.info("shape after padded: %s, %s", images.shape, masks.shape)

    # images = normalize(images)
    # test_data(images, masks, 1679)
    data_provider = image_util.SimpleDataProvider(images, masks, channels=3)
    logging.info("data_provider.channels: %s, data_provider.n_class: %s", data_provider.channels, data_provider.n_class)

    # test_data_provider(data_provider)
    net = unet.Unet(channels=data_provider.channels,
                    n_class=data_provider.n_class,
                    cost='cross_entropy',
                    layers=LAYERS,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )
    batch_size = 8
    net.verification_batch_size = batch_size * 2
    training_iters = (images.shape[0]-1) / batch_size + 1
    logging.info("batch_size: %s, iters: %s", batch_size, training_iters)

    trainer = unet.Trainer(net, batch_size=batch_size, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.9, learning_rate=0.01))
    path = trainer.train(data_provider, "log/20180416-1",
                         training_iters=training_iters, epochs=20, display_step=2)

if __name__ == "__main__":
    initlog()
    main()
