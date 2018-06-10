from __future__ import division, print_function
import os
import numpy as np
from PIL import Image as pil_image
import matplotlib.pyplot as plt
import cv2
import glob


def generate_background_masks(image_dir, mask_dir, my_palette):
    kernel = np.ones((20, 20), 'uint8')

    for fullname in glob.glob(image_dir + '/*.jpg'):
        fname = fullname.split('/')[-1].split('.')[0]
        # read image as greyscale
        tma_img = cv2.imread(fullname, 0)
        # Gaussian filtering to remove noise
        blur = cv2.GaussianBlur(tma_img, (25, 25), 0)
        # Otsu thresholding (background should be assigned 0, tissue with 1)
        ret, img_thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # add padding to avoid weird borders afterwards
        bb = 100
        img_thres = cv2.copyMakeBorder(img_thres,bb,bb,bb,bb,cv2.BORDER_CONSTANT,value=0)
        # dilation to fill black holes
        img = cv2.dilate(img_thres, kernel, iterations=5)
        # followed by erosion to restore borders, eat up small objects
        img = cv2.erode(img, kernel, iterations=10)
        # then dilate again
        img = cv2.dilate(img, kernel, iterations=5)
        # crop to restore original image
        ws = np.array(img)[bb:-bb, bb:-bb]

        mask = np.zeros((tma_img.shape[0], tma_img.shape[1]), dtype='uint8')
        mask[ws == 0] = 4
        mask[ws == 255] = 0
        hm = pil_image.fromarray(mask.astype('uint8'), 'P')
        hm.putpalette(my_palette)
        mask_name = os.path.join(mask_dir, 'mask_' + fname + '.png')
        hm.save(mask_name)


def main():
    my_palette = [0, 255, 0,  # benign is green
                  0, 0, 255, # Gleason 3 is blue
                  255, 255, 0, # Gleason 4 is yellow
                  255, 0, 0, # Gleason 5 is red
                  255, 255, 255] # ignore class is white

    pref = '/data3/eirini/dataset_TMA'
    image_dir = os.path.join(pref, 'TMA_images')
    mask_dir = os.path.join(pref, 'tissue_masks')
    os.makedirs(mask_dir)
    generate_background_masks(image_dir, mask_dir, my_palette)


if __name__ == '__main__':
    main()