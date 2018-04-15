
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
	image_part = np.reshape(image_part, (1,)+image_part.shape)
	image_part_pad = padding_array(image_part, offset, default_val=0)
	pred = net.infer(image_part_pad)
	pred = pred[0,...,1]
	pred = pred > 0.5
	pred = crop_to_shape(pred, original_shape)
	logging.info("original_shape: %s, pad.shape: %s, pred.shape: %s, crop_pred.shape: %s",
		original_shape, image_part_pad.shape, pred.shape, crop_pred.shape)

	return pred



def infer_test():
	pkl_fname = "data/preprocess/stage1_test_set.pkl"
    with open(pkl_fname, "rb") as f:
        ds = pickle.load(f)

    net = unet.Unet(channels=data_provider.channels,
                    n_class=data_provider.n_class,
                    cost='cross_entropy',
                    layers=LAYERS,
                    features_root=64,
                    cost_kwargs=dict(regularizer=0.001),
                    )
    net.load_weight("log/20180414/model.cpkt")

    for image, value in ds.items():
    	image_part = value[0]['img']
    	mask = infer_part(image_part)

    	fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,5))
        ax[0].imshow(image_part, aspect="auto")
        ax[2].imshow(mask, aspect="auto")
        ax[0].set_title("Input")
        ax[1].set_title("Ground truth")
        ax[2].set_title("Prediction")
        fig.tight_layout()
        plt.show()



if __name__ == "__main__":
    initlog()
    main()