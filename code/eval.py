
from mlog import initlog
import glob
import os
import numpy as np
import logging
import cv2
import math
import cPickle as pickle
import matplotlib.pylab as plt

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util
from train import get_dataset, unet_size, padding_array


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[0] - shape[0])//2
    offset1 = (data.shape[1] - shape[1])//2
    return data[offset0:(-offset0), offset1:(-offset1)]

def calc_dice(pred, gt):
    print(pred.shape, gt.shape)
    gt_cropped = crop_to_shape(gt, pred.shape)
    print(pred.shape, gt.shape, gt_cropped.shape)
    pred = pred.astype(np.bool)
    tp = np.sum(np.logical_and(pred==True, gt_cropped==True))
    fp = np.sum(np.logical_and(pred==True, gt_cropped==False))
    fn = np.sum(np.logical_and(pred==False, gt_cropped==True))
    
    print(tp,fp,fn)
    dice = tp / float(tp+fp+fn)
    return dice
    



def main():
	LAYERS = 3
    pkl_fname = "data/preprocess/stage1_train_set.pkl"
    images, masks = get_dataset(pkl_fname)
    logging.info("read train set: %s, %s", images.shape, masks.shape)
    logging.info("image:[%s, %s], mask:[%s, %s]", np.max(images), np.min(images), np.max(masks), np.min(masks))

    pred_size, offset = unet_size(256, LAYERS)
    logging.info("pred_size: %d, offset: %d", pred_size, offset)
    images = padding_array(images, offset, default_val=0.0)
    masks = padding_array(masks, offset, default_val=False)
    logging.info("shape after padded: %s, %s", images.shape, masks.shape)
	pkl_fname = "data/preprocess/stage1_train_set_padding.pkl"
    with open(pkl_fname, "wb") as f:
    	pickle((images, masks), f, protocol=pickle.HIGHEST_PROTOCOL)


    # test_data(images, masks, 1679)
    data_provider = image_util.SimpleDataProvider(images, masks)
    logging.info("data_provider.channels: %s, data_provider.n_class: %s", data_provider.channels, data_provider.n_class)

    # test_data_provider(data_provider)
    net = unet.Unet(channels=data_provider.channels,
                    n_class=data_provider.n_class,
                    cost='cross_entropy',
                    layers=LAYERS,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )

	x_test, y_test = data_provider(1)
	prediction = net.predict("log/20180414/model.cpkt", x_test)
	mask = prediction[0,...,1] > 0.5
	img_cropped = crop_to_shape(x_test[0,...,0], mask.shape)
	gt = y_test[0,...,1]
	gt_cropped = crop_to_shape(gt, mask.shape)

	fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
	ax[0].imshow(img_cropped, aspect="auto")

	ax[1].imshow(gt_cropped, aspect="auto")
	ax[2].imshow(mask, aspect="auto")
	ax[0].set_title("Input")
	ax[1].set_title("Ground truth")
	ax[2].set_title("Prediction")
	fig.tight_layout()
	logging.info("dice: %f", calc_dice(mask, gt_cropped))


if __name__ == "__main__":
    initlog()
    main()