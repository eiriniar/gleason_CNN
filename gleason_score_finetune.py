from __future__ import division, print_function
import os
import sys
import pickle
import glob
import numpy as np
import pandas as pd

# possible neural network architectures
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from cnn_finetune.densenet121 import densenet121_model
from cnn_finetune.custom_layers.scale_layer import Scale

# and corresponding pre-processing
from keras.applications.imagenet_utils import preprocess_input
# VGG, ResNet, DenseNet work in [0, 255] range
# Inception, MobileNet work in [-1, 1] range

# Keras layers and utils
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D
from keras.layers import Activation, Reshape
from keras import optimizers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from utils.keras_utils import balanced_generator

from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix


def get_filenames_and_classes(csv_path):
    df = pd.read_csv(csv_path, sep='\t', index_col=0)
    filenames, classes = [], []
    for base_name, primary_grade, sec_grade in zip(df.index, df.iloc[:,0], df.iloc[:,1]):
        primary_grade, sec_grade = int(primary_grade), int(sec_grade)
        filenames.append(base_name)
        classes.append(np.array([primary_grade, sec_grade]).reshape(1,2))
    classes = np.vstack(classes)
    return filenames, classes


def train_network(net='MobileNet_50'):

    # specify a directory to store model weights
    mpath = '/data3/eirini/gleason_CNN/models/finetune_%s' % net
    if not os.path.exists(mpath):
        os.makedirs(mpath)

    # specify the input directory, where image patches live
    prefix = '/data3/eirini/dataset_TMA'
    patch_dir = os.path.join(prefix, 'train_validation_patches_750')


    init_dim = 250 if net != 'InceptionV3' else 350
    dim = 224 if net != 'InceptionV3' else 299
    target_size = (dim, dim)
    input_shape = (target_size[0], target_size[1], 3)
    bs = 32

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # classes
    class_labels = ['benign', 'gleason3', 'gleason4', 'gleason5']
    n_class = len(class_labels)

    # training set
    train_filenames, train_classes = [], []
    for tma in ['ZT199', 'ZT204', 'ZT111']:
        csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
        new_filenames, new_classes = get_filenames_and_classes(csv_path)
        train_filenames += new_filenames
        train_classes.append(new_classes)
    train_classes = np.vstack(train_classes)

    # validation set
    tma = 'ZT76'
    csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
    val_filenames, val_classes = get_filenames_and_classes(csv_path)

    # test set
    tma = 'ZT80'
    csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
    test_filenames, test_classes = get_filenames_and_classes(csv_path)

    # total number of TMA spots in training set
    N = len(train_filenames)
    print('Total training TMAs: %d' % N)

    # print('training set:')
    # print(Counter(np.sum(train_classes, axis=1)))
    # print('validation set:')
    # print(Counter(np.sum(val_classes, axis=1)))

    # customize the pre-processing
    if net.startswith('MobileNet') or net.startswith('Inception'):
        preprocess_mode = 'tf'
    else:
        preprocess_mode = 'caffe'

   # define data generators
    train_batches = balanced_generator(patch_dir, filenames=train_filenames+train_filenames,
                                      classes=np.hstack([train_classes[:,0], train_classes[:,1]]),
                                      input_dim=init_dim, crop_dim=dim, batch_size=bs,
                                      data_augmentation=True, save_to_dir=None, add_classes=True,
                                      preprocess_mode=preprocess_mode)
    valid_batches = balanced_generator(patch_dir, filenames=val_filenames,
                                      classes=val_classes[:,0],
                                      input_dim=init_dim, crop_dim=dim, batch_size=bs,
                                      data_augmentation=False, save_to_dir=None, add_classes=True,
                                      preprocess_mode=preprocess_mode)
    model_weights = os.path.join(mpath, 'model_{epoch:02d}.h5')

    # get the model architecture
    print('Training %s ...' % net)
    if net == 'VGG16':
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg')
    elif net == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg')
    elif net == 'InceptionV3':
        # original size: 299x299
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg')
    elif net == 'MobileNet_100':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg',
                               alpha=1.0, depth_multiplier=1, dropout=.2)
    elif net == 'MobileNet_75':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg',
                               alpha=.75, depth_multiplier=1, dropout=.2)
    elif net == 'MobileNet_50':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg',
                               alpha=.5, depth_multiplier=1, dropout=.2)
    elif net == 'MobileNet_25':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg',
                               alpha=.25, depth_multiplier=1, dropout=.2)
    elif net == 'DenseNet121':
        tf_weights = 'cnn_finetune/imagenet_models/densenet121_weights_tf.h5'
        base_model = densenet121_model(dim, dim, 3, dropout_rate=0.2, weights_path=tf_weights)
    else:
        print('Unknown model, will train MobileNet instead.')
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(dim, dim, 3), pooling='avg',
                               alpha=.5, depth_multiplier=1, dropout=.2)


    # add top layer and compile model
    x_top = base_model.output
    x_out = Dense(n_class, name='output', activation='softmax')(x_top)
    model = Model(base_model.input, x_out)
    model.summary()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit_generator(
                generator=train_batches,
                steps_per_epoch=200,
                epochs=5,
                validation_data=valid_batches,
                validation_steps=100,
                verbose=1
    )

    # let the layers train freely
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics = ['accuracy'])
    checkpoint = ModelCheckpoint(filepath=model_weights, save_best_only=False, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.00001)

    history = model.fit_generator(
                generator=train_batches,
                steps_per_epoch=200,
                epochs=500,
                callbacks=[checkpoint, reduce_lr],
                validation_data=valid_batches,
                validation_steps=100,
                verbose=1
    )
    with open(os.path.join(mpath, 'history.pkl'), 'wb') as history_f:
        pickle.dump(history.history, history_f, protocol=2)



if __name__ == '__main__':
    net = sys.argv[1]
    train_network(net)



