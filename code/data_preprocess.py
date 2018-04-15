
from mlog import initlog
import glob
import os
import numpy as np
import logging
import cv2
import math
import cPickle as pickle


def read_image_mask(img_folder, has_mask=True):
    img_fname = glob.glob(os.path.join(img_folder, "images", "*.png"))[0]
    img = cv2.imread(img_fname)

    if has_mask:
        mask_fnames = glob.glob(os.path.join(img_folder, "masks", "*.png"))
        masks = np.array([cv2.imread(x)[:, :, 0] for x in mask_fnames])
        mask = (np.sum(masks, axis=0) > 0).astype(np.uint8)
    else:
        mask = None

    return img, mask


def calc_parts(x, s):
    n_parts = int(math.ceil(x/float(s)))
    parts = []
    for i in range(n_parts):
        if i == n_parts-1:
            p = x - s
        else:
            p = i * s
        parts.append(p)

    return parts


def split_parts(img, mask, s=(256,256)):
    if mask is not None:
        assert(img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1])

    if img.shape[0] < s[0] or img.shape[1] < s[1]:
        raise ValueError("image shape %s, is less than %s" % (img.shape, s))

    h_parts = calc_parts(img.shape[0], s[0])
    w_parts = calc_parts(img.shape[1], s[1])

    img_parts = []
    mask_parts = []
    parts = []

    for h in h_parts:
        for w in w_parts:
            img_parts.append(img[h:h+s[0], w:w+s[1]])
            if mask is not None:
                mask_parts.append(mask[h:h+s[0], w:w+s[1]])
            else:
                mask_parts.append(None)
            parts.append([h, w])

    return img_parts, mask_parts, parts



def get_dataset(folder, has_mask=True, n_samples=None, s=(256,256)):
    imgs_folders = glob.glob(os.path.join(folder, "*"))[:n_samples]
    ds = {}

    k = 0
    for i, f in enumerate(imgs_folders):
        img, mask = read_image_mask(f, has_mask)
        img = img[:, :, 0]    # for gray scale
        img_parts, mask_parts, parts = split_parts(img, mask, s)
        data = [{'pos': x[2], 'img': x[0], 'mask': x[1], 'raw_shape': img.shape} for x in zip(img_parts, mask_parts, parts)]
        img_id = os.path.basename(f)
        ds[img_id] = data
        k += len(data)
        logging.info("Precess folder: %s, shape: %s, parts: %s", f, img.shape, len(data))

    logging.info("totol read, image: %d parts: %s", len(imgs_folders), k )
    return ds

def do_trainset():
    data_dir = "data/datasets/stage1_train"
    ds = get_dataset(data_dir, n_samples=None)

    pkl_fname = "data/preprocess/stage1_train_set.pkl"
    with open(pkl_fname, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

def do_testset():
    data_dir = "data/datasets/stage1_test"
    ds = get_dataset(data_dir, has_mask=False, n_samples=None, s=(128,128))

    pkl_fname = "data/preprocess/stage1_test_set.pkl"
    with open(pkl_fname, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    initlog()
    do_testset()
