from __future__ import print_function

import os
import numpy as np
import pandas as pd
import scipy.misc
from PIL import Image
from collections import Counter


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
    def __init__(self, path_images, path_masks, path_patches):
        self.path_images = path_images
        self.path_patches = path_patches
        self.path_masks = path_masks
        self.palette = [0, 255, 0, # benign is green (index 0)
                        0, 0, 255, # Gleason 3 is blue (index 1)
                        255, 255, 0, # Gleason 4 is yellow (index 2)
                        255, 0, 0, # Gleason 5 is red (index 3)
                        255, 255, 255] # ignore class is white (index 4)
        if not os.path.exists(self.path_masks):
            os.makedirs(self.path_masks)
        if not os.path.exists(self.path_patches):
            os.makedirs(self.path_patches)


    def create_annotated_patches(self, tma_pref, patch_size, n_class=4):
        # loop over images
        for filename in os.listdir(self.path_images):
            if filename.startswith(tma_pref):
                name = filename.rstrip('.jpg')
                fullmask = os.path.join(self.path_masks, 'mask_' + name + '.png')

                # if an annotation exists
                if os.path.exists(fullmask):
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
        mask_dir_1 = os.path.join(self.path_masks, 'Gleason_masks_test_pathologist1')
        mask_dir_2 = os.path.join(self.path_masks, 'Gleason_masks_test_pathologist2')

        # open a file for writing down the grades
        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s' % ('patch_name', 'grade_1', 'grade_2'), file=f_out)

        # loop over images
        for filename in os.listdir(self.path_images):
            if filename.startswith(tma_pref):
                name = filename.rstrip('.jpg')
                fullmask_1 = os.path.join(mask_dir_1, 'mask1_' + name + '.png')
                fullmask_2 = os.path.join(mask_dir_2, 'mask2_' + name + '.png')

                # if an annotation exists by both pathologists
                if os.path.exists(fullmask_1) and os.path.exists(fullmask_2):
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
                                # write down the patch labels
                                patch_name = '%s_patch_%d' % (name, j)
                                print('%s\t%d\t%d' % (patch_name, grade_1, grade_2), file=f_out)
        f_out.close()

    def count_gleason(self, tma_pref, csv_file, n_class=4):
        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s' % ('TMA_spot', 'class_primary', 'class_secondary'), file=f_out)

        # loop through all masks
        for filename in os.listdir(self.path_masks):
            if filename.startswith('mask_%s' % tma_pref):
                key = filename.lstrip('mask_').rstrip('.png')
                full_path = os.path.join(self.path_masks, filename)
                primary_score, secondary_score = sum_up_gleason(full_path, n_class)
                print('%s\t%d\t%d' % (key, primary_score, secondary_score), file=f_out)
        f_out.close()

    def count_gleason_joint(self, csv_file, n_class=4):
        mask_dir_1 = os.path.join(self.path_masks, 'Gleason_masks_test_pathologist1')
        mask_dir_2 = os.path.join(self.path_masks, 'Gleason_masks_test_pathologist2')

        f_out = open(csv_file, 'w')
        print('%s\t%s\t%s\t%s\t%s' % ('TMA_spot', 'patho1_class_primary', 'patho1_class_secondary',
                                      'patho2_class_primary', 'patho2_class_secondary'), file=f_out)
        # loop through all masks
        for filename in os.listdir(mask_dir_1):
            if filename.endswith('.png'):
                key = filename.lstrip('mask1_').rstrip('.png')
                # read the first pathologist's annotation
                full_path_1 = os.path.join(mask_dir_1, filename)
                primary_1, secondary_1 = sum_up_gleason(full_path_1, n_class)
                # read the second pathologist's annotation
                full_path_2 = os.path.join(mask_dir_2, 'mask2_'+key+'.png')
                primary_2, secondary_2 = sum_up_gleason(full_path_2, n_class)
                print('%s\t%d\t%d\t%d\t%d' % (key, primary_1, secondary_1, primary_2, secondary_2), file=f_out)
        f_out.close()



def main():
    prefix  = '/data3/eirini/dataset_TMA'

    # directory containing all TMA spot images
    path_images = os.path.join(prefix, 'TMA_images')

    # directory containing training labels (Gleason annotation masks by first pathologist)
    path_train_masks = os.path.join(prefix, 'Gleason_masks_train')

    # directory containing test labels (Gleason annotation masks by second pathologist)
    path_test_masks = os.path.join(prefix, 'Gleason_masks_test')

    # directory where summary intermediate files are saved
    tma_info_path = os.path.join(prefix, 'tma_info')
    if not os.path.exists(tma_info_path):
        os.makedirs(tma_info_path)

    # TMAs used for training/validation
    tma_names = ['ZT76', 'ZT111', 'ZT199', 'ZT204']
    tma_prefixes = ['ZT76_39', 'ZT111_4', 'ZT199_1', 'ZT204_6']
    patch_size = 750
    path_patches = os.path.join(prefix, 'train_validation_patches_%d' % patch_size)

    # create patches and patch labels (training/validation sets)
    for tma_name, tma_prefix in zip(tma_names, tma_prefixes):
        proc = ImageProcessor(path_images, path_train_masks, path_patches)
        proc.create_annotated_patches(tma_prefix, patch_size=patch_size)
        # write down a summary (primary and secondary Gleason patterns) of the annotation
        csv_file = os.path.join(tma_info_path, '%s_gleason_scores.csv' % tma_name)
        proc.count_gleason(tma_prefix, csv_file)

    # TMAs used for testing
    tma_name, tma_prefix = 'ZT80', 'ZT80_38'
    path_test_patches = os.path.join(prefix, 'test_patches_%d' % patch_size)

    # create patches for the test cohort (only use patches annotated by both pathologists)
    joint_proc = ImageProcessor(path_images, path_test_masks, path_test_patches)
    patch_file = os.path.join(tma_info_path, 'ZT80_patch_grades.csv')
    joint_proc.create_joint_patches(tma_prefix, patch_size=patch_size, csv_file=patch_file)
    csv_file = os.path.join(tma_info_path, '%s_gleason_scores.csv' % tma_name)
    joint_proc.count_gleason_joint(csv_file)



if __name__ == "__main__":
    main()

