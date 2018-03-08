from __future__ import print_function

import os
import subprocess
import glob
import xmltodict
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.patches as mplPatches
import numpy as np
import scipy.misc
import pandas as pd
from itertools import product
from PIL import Image
import cv2
from collections import Counter



def parse_xml(xml_file):
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
        if doc['object-stream']['TMAspot']['includingAreas'] is not None:
            roi = doc['object-stream']['TMAspot']['includingAreas']['java.awt.Polygon']
            roi_label = doc['object-stream']['TMAspot']['nuclei']['TMApoint']
            # handle a single ROI or a list of ROIs in a uniform way
            if not isinstance(roi, list):
                roi = [roi]
                roi_label = [roi_label]
        else:
            roi, roi_label = [], []
    return roi, roi_label

class ROIreader:
    ''' A class for reading xml annotations from TMARKER '''

    def __init__(self):
        # annotated images
        self.image_names = []
        # corresponding ROIs
        self.roi_list = []
        # and ROI labels
        self.roi_labels = []

    def read_roi_xml(self, annotation_path):
        # process only images for which annotations exist
        for ii, path_xml in enumerate(glob.glob(annotation_path+'/*.xml')):
            annot_xml = path_xml.split('/')[-1]
            self.image_names.append(annot_xml.split('.')[0])
            roi, roi_label = parse_xml(path_xml)
            self.roi_list.append(roi)
            self.roi_labels.append(roi_label)


def path_from_roi(roi):
    j_points = roi['xpoints']['int']
    i_points = roi['ypoints']['int']
    coord = np.array(list(zip(i_points, j_points))).astype(np.int)
    # discard the (0, 0) coordinate if present (should not have been there) 
    keep = np.sum(coord, axis=1) > 0
    bbPath = mplPath.Path(coord[keep], closed=True)
    return bbPath

def get_image_patch_coord(size_x, size_y, patch_size):
    """ Generates overlapping patches from an image. """
    step = patch_size // 2
    nx = (size_x-patch_size) // step + 1
    ny = (size_y-patch_size) // step + 1
    patch_coord = np.ndarray(shape=(nx*ny, 2), dtype=np.int32)
    i = 0
    for y in range(0, size_y-patch_size, step):
        for x in range(0, size_x-patch_size, step):
            patch_coord[i] = [x, y]
            i += 1
    return patch_coord[:i]

def get_image_mask(roi_list, roi_labels, img_dim, ignore_index=4):
    mask = ignore_index * np.ones((img_dim, img_dim), dtype='uint8')
    coordinates = list(product(xrange(img_dim), xrange(img_dim)))

    # first we sort the list of ROIs, so that the biggest one is annotated first
    # this is done to handle nested ROI annotations
    if len(roi_list) > 1:
        roi_sizes = []
        for roi in roi_list:
            bbPath = path_from_roi(roi)
            roi_sizes.append(np.sum(bbPath.contains_points(coordinates)))
        sorted_idx = np.argsort(roi_sizes)[::-1]
    else:
        sorted_idx = [0]

    for i in sorted_idx:
        roi = roi_list[i]
        roi_label = matching_label(roi, roi_labels, roi_list)
        grade = int(roi_label['staining']) + 1
        bbPath = path_from_roi(roi)
        # make sure that the ROI label is the correct one
        assert bbPath.contains_point((int(roi_label['y']), int(roi_label['x'])))
        mask[bbPath.contains_points(coordinates).reshape(img_dim, img_dim)] = grade
    return mask

def matching_label(curr_roi, roi_labels, all_rois=[]):
    bbPath = path_from_roi(curr_roi)
    matching = []
    for roi_label in roi_labels:
        if bbPath.contains_point((int(roi_label['y']), int(roi_label['x']))):
            matching.append(roi_label)

    # there should be at least one matching label for the current ROI
    assert len(matching) > 0
    matching_label = matching[0]

    # handle the case of nested ROIs
    # by determining which label is not contained into any other ROI
    if len(matching) > 1:
        for roi_label in matching:
            label_count = 0
            for roi in all_rois:
                bbPath = path_from_roi(roi)
                if bbPath.contains_point((int(roi_label['y']), int(roi_label['x']))):
                    label_count += 1
            if label_count == 1:
                matching_label = roi_label
                break
    return matching_label

def load_mask(fullmask):
    mask = Image.open(fullmask)
    return np.array(mask)

def get_patch_label(i0, j0, patch_size, mask, n_class=4):
    ws = patch_size // 3
    central_grades = mask[(i0+ws):(i0+2*ws), (j0+ws):(j0+2*ws)]
    grades_found = np.unique(central_grades)
    grades_found = grades_found[grades_found < n_class]
    grade = grades_found[0] if len(grades_found) == 1 else n_class
    return grade


# adapted from: http://robotics.usc.edu/~ampereir/wordpress/?p=626
def SaveFigureAsImage(fileName, fig=None, **kwargs):
    fig_size = fig.get_size_inches()
    w, h = fig_size[0], fig_size[1]
    if kwargs.has_key('orig_size'):
        w, h = kwargs['orig_size']
        w2, h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fileName, format='pdf')

def draw_annot(ax, img, roi_list, roi_labels, colors=['b','y','r'], lw=15):
    ax.imshow(img, origin='upper')
    for roi in roi_list:
        roi_label = matching_label(roi, roi_labels, roi_list)
        curr_grade = int(roi_label['staining'])
        bbPath = path_from_roi(roi)
        x, y = zip(*bbPath.vertices)
        ax.plot(y, x, '%so-' % colors[curr_grade], linewidth=lw)
    return ax

def annotate_img(img, roi_list, roi_labels, name, saving_name, colors=['b','y','r'], lw=20):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)
    fig.suptitle(name, fontsize=20)
    draw_annot(ax, img, roi_list, roi_labels, colors=colors, lw=lw)
    SaveFigureAsImage(saving_name, plt.gcf())
    plt.clf()
    plt.close()

def open_jpg(img):
    return scipy.misc.imread(img, mode='RGB')

def is_too_white(img, limit=190):
    return np.mean(img) > limit

def save_patch(patch, saving_name):
    image = Image.fromarray(patch)
    image.save(saving_name)

def sum_up_gleason(maskfile, n_class=4):
    # read the mask and count the grades
    mask = load_mask(maskfile)
    c = Counter(mask.flatten())
    grade_count = np.zeros(n_class, dtype=int)
    for i in range(n_class):
        grade_count[i] = c[i]

    # get the max and second max scores and write them to file
    idx = np.argsort(grade_count)
    primary_score = idx[-1]
    secondary_score = idx[-2]
    if np.sum(grade_count == 0) == 3:
        secondary_score = primary_score
    return primary_score, secondary_score


class ImageProcessor():
    def __init__(self, path_images, path_masks, path_patches, path_annot):
        self.path_images = path_images
        self.path_patches = path_patches
        self.path_masks = path_masks
        self.path_annot = path_annot
        self.palette = [0, 255, 0, # benign is green (index 0)
                        0, 0, 255, # Gleason 3 is blue (index 1)
                        255, 255, 0, # Gleason 4 is yellow (index 2)
                        255, 0, 0, # Gleason 5 is red (index 3)
                        255, 255, 255] # ignore class is white (index 4)
        if not os.path.exists(self.path_masks):
            os.makedirs(self.path_masks)
        if not os.path.exists(self.path_patches):
            os.makedirs(self.path_patches)
        if self.path_annot is not None:
            if not os.path.exists(self.path_annot):
                os.makedirs(self.path_annot)


    def draw_pdf_annotations(self, cl):
        i = 0
        for filename in os.listdir(self.path_images):
            if filename.endswith(".jpg"):
                fullname = os.path.join(self.path_images, filename)
                name = filename.split('.')[0]
                if name in cl.image_names:
                    idx = cl.image_names.index(name)
                    img = open_jpg(fullname)

                    # get the cancer ROI from the annotations
                    roi_coord = cl.roi_list[idx]
                    roi_label = cl.roi_labels[idx]
                    print(name)
                    print('Number of ROIs: %d\n' % len(roi_coord))
                    i += 1
                    saving_name = os.path.join(self.path_annot, '%s.pdf' % name)
                    annotate_img(img, roi_coord, roi_label, name, saving_name)
        print('Number of TMA spots considered: %d' % i)


    def create_masks(self, cl):
        i = 0
        for filename in os.listdir(self.path_images):
            if filename.endswith(".jpg"):
                fullname = os.path.join(self.path_images, filename)
                name = filename.split('.')[0]
                if name in cl.image_names:
                    idx = cl.image_names.index(name)
                    img = open_jpg(fullname)

                    # get the cancer ROI from Kim's annotations
                    roi_coord = cl.roi_list[idx]
                    roi_label = cl.roi_labels[idx]
                    print(name)
                    print('Number of ROIs: %d\n' % len(roi_coord))
                    # if there is an annotation on the image, draw the corresponding mask
                    if len(roi_coord) > 0:
                        i += 1
                        image_mask = get_image_mask(roi_coord, roi_label, img.shape[0], ignore_index=4)
                        saving_name = os.path.join(self.path_masks, 'mask_%s.png' % name)
                        image = Image.fromarray(image_mask, 'P')
                        image.putpalette(self.palette)
                        image.save(saving_name)
                        # test that it worked
                        #im = Image.open(saving_name)
                        #indexed = np.array(im)
                        #print(indexed.shape)
                        #print(np.unique(indexed))
        print('Number of TMA spots considered: %d' % i)


    def create_annotated_patches(self, tma_pref, patch_size, n_class=4):
        # loop over images
        for filename in os.listdir(self.path_images):
            if filename.startswith(tma_pref):
                name = filename.rstrip('.jpg')
                maskname = 'mask_' + name + '.png'
                fullmask = os.path.join(self.path_masks, maskname)

                # if an annotation exists
                if os.path.exists(fullmask):
                    print(name)
                    subdir = os.path.join(self.path_patches, name)
                    os.makedirs(subdir)

                    # read the image
                    fullname = os.path.join(self.path_images, filename)
                    img = open_jpg(fullname)
                    size_y, size_x = img.shape[0], img.shape[1]
                    # load the mask
                    mask = load_mask(fullmask)

                    patch_coord = get_image_patch_coord(size_x, size_y, patch_size)
                    for j, (i_0, j_0) in enumerate(patch_coord):
                        patch = img[i_0:i_0+patch_size, j_0:j_0+patch_size]
                        grade = get_patch_label(i_0, j_0, patch_size, mask)
                        # if the patch was annotated with a single Gleason grade
                        # and does not contain mostly background,
                        # then save it
                        if (grade < n_class) and (not is_too_white(patch, limit=180)):
                            patch_name = os.path.join(subdir, '%s_patch_%d_class_%d.jpg' % (name, j, grade))
                            save_patch(patch, patch_name)


    def create_joint_patches(self, tma_pref, patch_size, csv_file, n_class=4):
        # subdirectories, one for each pathologist
        subdir_patho_1 = os.path.join(self.path_patches, 'patho_1')
        subdir_patho_2 = os.path.join(self.path_patches, 'patho_2')
        os.makedirs(subdir_patho_1)
        os.makedirs(subdir_patho_2)
        mask_dir_1 = os.path.join(self.path_masks, 'kim_ZT80_Gleason_masks')
        mask_dir_2 = os.path.join(self.path_masks, 'jan_ZT80_Gleason_masks')

        # open a file for writing down the grades
        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s' % ('patch_name', 'grade_1', 'grade_2'), file=f_out)

        # loop over images
        for filename in os.listdir(self.path_images):
            if filename.startswith(tma_pref):
                name = filename.rstrip('.jpg')
                maskname = 'mask_' + name + '.png'
                fullmask_1 = os.path.join(mask_dir_1, maskname)
                fullmask_2 = os.path.join(mask_dir_2, maskname)

                # if an annotation exists by both pathologists
                if os.path.exists(fullmask_1) and os.path.exists(fullmask_2):
                    print(name)
                    dir_1 = os.path.join(subdir_patho_1, name)
                    dir_2 = os.path.join(subdir_patho_2, name)
                    os.makedirs(dir_1)
                    os.makedirs(dir_2)

                    # read the image
                    fullname = os.path.join(self.path_images, filename)
                    img = open_jpg(fullname)
                    size_y, size_x = img.shape[0], img.shape[1]
                    # load the masks
                    mask_1 = load_mask(fullmask_1)
                    mask_2 = load_mask(fullmask_2)

                    patch_coord = get_image_patch_coord(size_x, size_y, patch_size)
                    for j, (i_0, j_0) in enumerate(patch_coord):
                        grade_1 = get_patch_label(i_0, j_0, patch_size, mask_1)
                        grade_2 = get_patch_label(i_0, j_0, patch_size, mask_2)
                        # if the patch was annotated by both pathologists
                        if (grade_1 < n_class) and (grade_2 < n_class):
                            patch = img[i_0:i_0+patch_size, j_0:j_0+patch_size]
                            # and it does not contain mostly background
                            if not is_too_white(patch, limit=180):
                                # save the patch
                                patch_name_1 = os.path.join(dir_1, '%s_patch_%d_class_%d.jpg' % (name, j, grade_1))
                                patch_name_2 = os.path.join(dir_2, '%s_patch_%d_class_%d.jpg' % (name, j, grade_2))
                                save_patch(patch, patch_name_1)
                                save_patch(patch, patch_name_2)
                                # write down the labels
                                patch_name = '%s_patch_%d' % (name, j)
                                print('%s\t%d\t%d' % (patch_name, grade_1, grade_2), file=f_out)
        f_out.close()


    def count_gleason(self, tma_pref, csv_file, n_class=4):
        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s' % ('TMA_spot', 'class_primary', 'class_secondary'), file=f_out)

        # loop through all masks
        for filename in os.listdir(self.path_masks):
            if filename.startswith('mask_%s' % tma_pref):
                print(filename)
                key = filename.lstrip('mask_').rstrip('.png')
                full_path = os.path.join(self.path_masks, filename)
                primary_score, secondary_score = sum_up_gleason(full_path, n_class)
                print('%s\t%d\t%d' % (key, primary_score, secondary_score), file=f_out)
        f_out.close()


    def count_gleason_joint(self, csv_file, n_class=4):
        mask_dir_1 = os.path.join(self.path_masks, 'kim_ZT80_Gleason_masks')
        mask_dir_2 = os.path.join(self.path_masks, 'jan_ZT80_Gleason_masks')

        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s\t%s\t%s' % ('TMA_spot', 'kim_class_primary', 'kim_class_secondary',
                                      'jan_class_primary', 'jan_class_secondary'), file=f_out)
        # loop through all masks
        for filename in os.listdir(mask_dir_1):
            if filename.endswith('.png') and os.path.exists(os.path.join(mask_dir_2, filename)):
                print(filename)
                key = filename.lstrip('mask_').rstrip('.png')
                # read Kim's annotation
                full_path_1 = os.path.join(mask_dir_1, filename)
                primary_1, secondary_1 = sum_up_gleason(full_path_1, n_class)
                # read Jan's annotation
                full_path_2 = os.path.join(mask_dir_2, filename)
                primary_2, secondary_2 = sum_up_gleason(full_path_2, n_class)
                print('%s\t%d\t%d\t%d\t%d' % (key, primary_1, secondary_1, primary_2, secondary_2), file=f_out)
        f_out.close()



def main():
    prefix  = '/data3/eiriniar/gleason_CNN/dataset_TMA'
    tma_info_path = os.path.join(prefix, 'tma_info')
    path_xml_annotations = os.path.join(prefix, 'xml_annotations')
    path_pdf_annotations = os.path.join(prefix, 'pdf_annotations')
    path_images = os.path.join(prefix, 'TMA_images')
    path_Gleason_masks = os.path.join(prefix, 'Gleason_masks')

    tma_names = ['ZT76', 'ZT111', 'ZT199', 'ZT204']
    tma_prefixes = ['ZT76_39', 'ZT111_4', 'ZT199_1', 'ZT204_6']
    patch_size = 750
    path_patches = os.path.join(prefix, 'train_validation_patches_%d' % patch_size)

    # list of spots marked as benign
    benign_list_path = os.path.join(prefix, 'tma_info', 'benign_spots.txt')
    benign_list = list(pd.read_csv(benign_list_path, header=None).iloc[:,0])


    ''' training and validation TMAs were annotated by Kim '''

    for tma_name, tma_prefix in zip(tma_names, tma_prefixes):
        path_tma_xml = os.path.join(path_xml_annotations, 'kim_%s_xml' % tma_name)
        path_tma_pdf = os.path.join(path_pdf_annotations, 'kim_%s_pdf' % tma_name)
        if not os.path.exists(path_tma_pdf):
            os.makedirs(path_tma_pdf)

        # read in annotated ROIs
        reader = ROIreader()
        reader.read_roi_xml(path_tma_xml)

        proc = ImageProcessor(path_images, path_Gleason_masks, path_patches, path_tma_pdf)

        # overlay Gleason annotations on TMA images
        proc.draw_pdf_annotations(reader)

        # store Gleason annotations as masks
        proc.create_masks(reader)

        # store tissue masks for benign TMA spots
        for tma_spot_name in benign_list:
            if tma_spot_name.startswith(tma_prefix):
                source = os.path.join(prefix, 'tissue_masks', 'mask_%s.png' % tma_spot_name)
                target = os.path.join(prefix, 'Gleason_masks', 'mask_%s.png' % tma_spot_name)
                if not os.path.exists(target):
                    subprocess.call('cp %s %s' % (source, target), shell=True)

        # create patches
        proc.create_annotated_patches(tma_prefix, patch_size=patch_size)

        csv_file = os.path.join(tma_info_path, '%s_gleason_scores.csv' % tma_name)
        proc.count_gleason(tma_prefix, csv_file)


    ''' test TMAs were annotated by Kim and Jan, we only keep the spots annotated by both '''

    tma_name, tma_prefix = 'ZT80', 'ZT80_38'
    test_prefix = os.path.join(prefix, 'inter_observer')
    if not os.path.exists(test_prefix):
        os.makedirs(test_prefix)
    path_test_patches = os.path.join(test_prefix, 'joint_test_patches_%d' % patch_size)

    for pathologist in ['kim', 'jan']:
        path_tma_xml = os.path.join(path_xml_annotations, '%s_%s_xml' % (pathologist, tma_name))
        path_tma_pdf = os.path.join(path_pdf_annotations, '%s_%s_pdf' % (pathologist, tma_name))
        if not os.path.exists(path_tma_pdf):
            os.makedirs(path_tma_pdf)

        # path for Gleason masks
        test_mask_dir = os.path.join(test_prefix, '%s_%s_Gleason_masks' % (pathologist, tma_name))
        if not os.path.exists(test_mask_dir):
            os.makedirs(test_mask_dir)

        # read in annotated ROIs
        reader = ROIreader()
        reader.read_roi_xml(path_tma_xml)
        proc = ImageProcessor(path_images, test_mask_dir, path_test_patches, path_tma_pdf)
        # overlay Gleason annotations on TMA images
        proc.draw_pdf_annotations(reader)

        # store Gleason annotations as masks
        proc.create_masks(reader)
        # copy benign tissue masks, if a spot has not been annotated
        for tma_spot_name in benign_list:
            if tma_spot_name.startswith(tma_prefix):
                source = os.path.join(prefix, 'tissue_masks', 'mask_%s.png' % tma_spot_name)
                target = os.path.join(test_mask_dir, 'mask_%s.png' % tma_spot_name)
                if not os.path.exists(target):
                    subprocess.call('cp %s %s' % (source, target), shell=True)

    # create joint patches for the test cohort
    joint_proc = ImageProcessor(path_images, test_prefix, path_test_patches, None)
    patch_file = os.path.join(test_prefix, 'ZT80_patch_grades.csv')
    joint_proc.create_joint_patches(tma_prefix, patch_size=patch_size, csv_file=patch_file)

    csv_file = os.path.join(tma_info_path, '%s_gleason_scores.csv' % tma_name)
    joint_proc.count_gleason_joint(csv_file)



if __name__ == "__main__":
    main()

