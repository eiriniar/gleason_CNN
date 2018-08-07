from __future__ import division, print_function
import os
import sys
import glob
import numpy as np
import pandas as pd

from utils.keras_utils import preprocess_input_tf, center_crop
from gleason_score_finetune import get_filenames_and_classes

import keras.backend as K
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import AveragePooling2D, Conv2D, UpSampling2D
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from PIL import Image as pil_image


PLOT_HEATMAPS = True
PLOT_CAM = True

# we are in test mode
K.set_learning_phase(0)


def pil_resize(img, target_size):
    hw_tuple = (target_size[1], target_size[0])
    if img.size != hw_tuple:
        img = img.resize(hw_tuple)
    return img

def clean_axis(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def customize_axis(axis, title):
    axis.set_title(title)
    axis.grid(False)
    clean_axis(axis)
    return axis

def plot_output(filenames, image_dir, annot_dir_1, annot_dir_2, tissue_mask_dir,
                model, tdim, outdir, n_class=4):

    palette = [0, 255, 0, # benign is green (index 0)
            0, 0, 255, # Gleason 3 is blue (index 1)
            255, 255, 0, # Gleason 4 is yellow (index 2)
            255, 0, 0, # Gleason 5 is red (index 3)
            255, 255, 255] # ignore class is white (index 4)

    def to_rgb(x):
        tdim = x.shape[0]
        a = np.zeros((tdim, tdim, 3), dtype='uint8')
        for i in range(tdim):
            for j in range(tdim):
                k = x[i, j]
                a[i,j,:] = palette[3*k:3*(k+1)]
        return a

    for fname in filenames:
        print(fname)
        full_imfile = os.path.join(image_dir, fname+'.jpg')
        # get network predictions as pixel-level heatmaps
        img = image.load_img(full_imfile, grayscale=False, target_size=(tdim, tdim))
        X = image.img_to_array(img)
        X = preprocess_input_tf(X)
        y_pred_prob = model.predict(X[np.newaxis,:,:,:], batch_size=1)[0]

        # get the first Gleaosn annotation mask
        mask_1 = os.path.join(annot_dir_1, 'mask1_'+fname+'.png')
        y1 = pil_image.open(mask_1)
        y1 = to_rgb(np.array(pil_resize(y1, target_size=(tdim, tdim))))

        # get the second Gleaosn annotation mask
        mask_2 = os.path.join(annot_dir_2, 'mask2_'+fname+'.png')
        y2 = pil_image.open(mask_2)
        y2 = to_rgb(np.array(pil_resize(y2, target_size=(tdim, tdim))))

        # get the tissue mask
        tissue_maskfile = os.path.join(tissue_mask_dir, 'mask_'+fname+'.png')
        tissue_mask = pil_image.open(tissue_maskfile)
        tissue_mask = np.array(pil_resize(tissue_mask, target_size=(tdim, tdim)))

        # plot heatmaps only at (predicted) tissue regions
        y_pred_prob[tissue_mask == n_class] = 0.

        # make the heatmap plots
        fig, ax = plt.subplots(2, 3)
        ax[0, 2].imshow(y1)
        customize_axis(ax[0, 2], 'Pathologist 1')
        ax[1, 2].imshow(y2)
        customize_axis(ax[1, 2], 'Pathologist 2')
        ax[0, 0].imshow(y_pred_prob[:,:,0], cmap=cm.jet, vmin=0, vmax=1)
        customize_axis(ax[0, 0], 'benign')
        ax[0, 1].imshow(y_pred_prob[:,:,1], cmap=cm.jet, vmin=0, vmax=1)
        customize_axis(ax[0, 1], 'Gleason 3')
        ax[1, 0].imshow(y_pred_prob[:,:,2], cmap=cm.jet, vmin=0, vmax=1)
        customize_axis(ax[1, 0], 'Gleason 4')
        im = ax[1, 1].imshow(y_pred_prob[:,:,3], cmap=cm.jet, vmin=0, vmax=1)
        customize_axis(ax[1, 1], 'Gleason 5')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout()
        figpath = os.path.join(outdir, '_'.join([fname, 'heatmap_output']) + '.pdf')
        plt.savefig(figpath, format='pdf')
        plt.clf()
        plt.close()

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def plot_cam(filenames, classes, model, outdir, init_dim=250, tdim=224):
    for fname, pred_class in zip(filenames, classes):
        img = image.load_img(fname, grayscale=False, target_size=(init_dim, init_dim))
        X = image.img_to_array(img)
        X = center_crop(X, center_crop_size=(tdim, tdim))
        # get a copy
        x_img = X.copy().astype('uint8')

        # prepare for the network
        X = preprocess_input_tf(X)
        y_pred = model.predict(X[np.newaxis,:,:,:], batch_size=1)[0]
        if y_pred[pred_class] > .9:
            h1, h2 = visualize_class_activation_map(model, X[np.newaxis,:,:,:], pred_class, x_img)
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(x_img)
            customize_axis(ax[0], 'original image')
            ax[1].imshow(h1)
            customize_axis(ax[1], 'CAM heatmap')
            ax[2].imshow(h2)
            customize_axis(ax[2], 'highlighted regions')
            plt.tight_layout()
            patch_name = fname.split('/')[-1].split('.')[0]
            figpath = os.path.join(outdir, '_'.join(['cam', patch_name]) + '.pdf')
            plt.savefig(figpath, format='pdf')
            plt.clf()
            plt.close()

 
def visualize_class_activation_map(model, x, pred_class, original_img, cam_thres=.5,
                                   alpha=.5, beta =.1):
    ''' Code adapted from https://github.com/jacobgil/keras-cam '''

    class_weights = np.squeeze(model.layers[-1].get_weights()[0])
    n_class = class_weights.shape[1]
    final_conv_layer = get_output_layer(model, 'conv_pw_13_relu')
    get_output = K.function([model.layers[0].input], \
                            [final_conv_layer.output])
    [conv_outputs] = get_output([x])
    conv_outputs = conv_outputs[0, :, :, :]

    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:-1])
    for i, w in enumerate(class_weights[:, pred_class]):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam /= np.max(cam)
    dim = original_img.shape[0]
    cam = cv2.resize(cam, (dim, dim), interpolation=cv2.INTER_CUBIC)

    heatmap_colored = np.uint8(cm.jet(cam)[..., :3] * 255)
    heatmap_colored[np.where(cam < cam_thres)] = 0
    img2 = original_img.copy()
    img2[np.where(cam < cam_thres)] = 1
    heatmap_colored = np.uint8(original_img * alpha + heatmap_colored * (1. - alpha))
    transparent = cv2.addWeighted(original_img, beta, img2, 1-beta, 0)
    return heatmap_colored, transparent


def main(prefix):

    init_dim = 250
    dim = 224

    # classes
    class_labels = ['benign', 'gleason3', 'gleason4', 'gleason5']
    n_class = len(class_labels)
    patch_dir = os.path.join(prefix, 'test_patches_750', 'patho_1')
    mask_dir_1 = os.path.join(prefix, 'Gleason_masks_test', 'Gleason_masks_test_pathologist1')
    mask_dir_2 = os.path.join(prefix, 'Gleason_masks_test', 'Gleason_masks_test_pathologist2')
    image_dir = os.path.join(prefix, 'TMA_images')
    tissue_mask_dir = os.path.join(prefix, 'tissue_masks')

    # load the test set
    tma = 'ZT80'
    csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
    test_filenames, test_classes = get_filenames_and_classes(csv_path)
    N = len(test_filenames)
    print('TMA spots in test cohort: %d' % N)

    # load the trained patch-level model
    model_weights = 'model_weights/MobileNet_Gleason_weights.h5'
    patch_model = load_model(model_weights,
                             custom_objects={'relu6': relu6,
                                             'DepthwiseConv2D': DepthwiseConv2D
                                             })
    # provide an output directory
    outdir = 'results'


    #################################################################################
    ## output pixel-level heatmaps
    #################################################################################
    
    if PLOT_HEATMAPS:
        w_out, b_out = patch_model.layers[-1].get_weights()
        w_out = w_out[np.newaxis,np.newaxis,:,:]

        # create a model for predicting on whole TMAs
        # rescaling factor is 3
        big_dim = 1024
        base_model = MobileNet(include_top=False, weights=None,
                               input_shape=(big_dim, big_dim, 3),
                               alpha=.5, depth_multiplier=1, dropout=.2)
        block_name = 'conv_pw_13_relu'
        x_input = base_model.get_layer(block_name).output

        # average pooling instead of global pooling
        x = AveragePooling2D((7, 7), strides=(1,1), padding='same', name='avg_pool_top')(x_input)
        x = Conv2D(n_class, (1, 1), activation='softmax', padding='same')(x)
        x_out = UpSampling2D(size=(32, 32), name='upsample')(x)
        model = Model(base_model.input, x_out)
        model.load_weights(model_weights, by_name=True)
        model.layers[-2].set_weights([w_out, b_out])
        model.summary()

        heatmap_dir = os.path.join(outdir, 'heatmaps')
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)
        plot_output(test_filenames, image_dir, mask_dir_1, mask_dir_2, tissue_mask_dir,
                    model, tdim=big_dim, outdir=heatmap_dir)


    #################################################################################
    ## class activation maps
    #################################################################################

    if PLOT_CAM:
        cam_dir = os.path.join(outdir, 'CAM')
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)

        cam_filenames, cam_classes= [], []
        for fname, test_class in zip(test_filenames, test_classes[:,0]):
            subdir = os.path.join(patch_dir, fname)
            patch_files = glob.glob(subdir + '/*class_%d.jpg' % test_class)
            cam_filenames += patch_files
            cam_classes += [test_class] * len(patch_files)
        plot_cam(cam_filenames, cam_classes, patch_model, cam_dir, init_dim=init_dim, tdim=dim)


if __name__ == '__main__':
    # provide the directory where the dataset lives 
    data_prefix = '/data3/eirini/dataset_TMA'
    main(data_prefix)










