import numpy as np
import nibabel as nib
from keras.models import Model
from keras.layers import Input, core, concatenate, Conv2D, UpSampling2D, MaxPooling2D
from config import *
import os


def normallize(datax):
    mean = np.mean(datax)
    std = np.std(datax)
    datax -= mean
    datax /= std
    return datax

def class_weighting(labelset, validation_ratio):
    labelset = labelset[0:int(len(labelset)*(1-validation_ratio))]
    S = labelset.shape
    class0 = (S[0]*S[1]*S[2]*S[3] - np.count_nonzero(labelset[:, :, :, :, 0] == 1))*1.0/np.count_nonzero(labelset[:, :, : , :, 0] == 1)
    class1 = (S[0]*S[1]*S[2]*S[3] - np.count_nonzero(labelset[:, :, :, :, 1] == 1))*1.0/np.count_nonzero(labelset[:, :, : , :, 0] == 1)
    class2 = (S[0]*S[1]*S[2]*S[3] - np.count_nonzero(labelset[:, :, :, :, 2] == 1))*1.0/np.count_nonzero(labelset[:, :, : , :, 0] == 1)

    return [class0, class1, class2]


def fuse_argmax(predict_matrix, image_size, axis=2):  # in shape (num, image_size, image_size, predict_class)
    predict = np.argmax(predict_matrix, axis=axis)
    seg = np.zeros(shape=predict.shape, dtype=np.float32)
    seg[np.where(predict == 2)] = 240.0
    seg[np.where(predict == 1)] = 140.0
    #seg[np.where(predict==0)] = 0.0
    return np.reshape(seg, (seg.shape[0], image_size, image_size))
    #return seg


def insert_zeroSlice(volume, original_shape, zero_index):
    index = zero_index[0]
    matrix = np.ones(original_shape, dtype=np.float32)
    for i in index:
        matrix[i] = 0
    count = 0
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i]) != 0:
            matrix[i] = volume[count]
            count +=1
    return matrix


def brain_reshape(predict_data, width, height, subsize=image_size):
    patch_num = width/subsize * height/subsize
    #print patch_num, predict_data.shape[0]/patch_num
    brain = np.ndarray(shape=(predict_data.shape[0]/patch_num, width, height), dtype=np.float32)
    brain_patch = np.ndarray(shape=(patch_num, predict_data.shape[0]/patch_num, subsize, subsize), dtype=np.float32)
    #print predict_data.shape, brain_patch.shape, brain.shape
    for i in range(patch_num):
        brain_patch[i] = predict_data[i*predict_data.shape[0]/patch_num:(i+1)*predict_data.shape[0]/patch_num,:,:]
    for row in range(brain.shape[1]/subsize):
        for col in range(brain.shape[2]/subsize):
            brain[:, row*subsize: (row+1)*subsize, col*subsize: (col+1)*subsize] = brain_patch[row*brain.shape[2]/subsize+col]
    return brain


def remove_padding(volume, extra_padx, extra_pady, x, y):
    return volume[:, extra_padx/2: extra_padx/2 + x, extra_pady/2: extra_pady/2 + y]


def attach_2_original_brain(volume, slice_track, size_track, direction, test=True):

    count = 0
    if test:
        object_number = 13  # testing data
        offset = 11
    else:
        object_number = 10  # training data
        offset = 1
    for i in range(object_number):
        name = 'subject-' + str(offset+i) + direction + '-label.img'
        print('attaching prediction volume ' + str(offset + i) + 'to original volume ( ' + name + ' )')
        subject = np.zeros(shape=(size_track[i][0], size_track[i][1], size_track[i][2]))
        S = slice_track[i][1] - slice_track[i][0]
        subject[slice_track[i][0]:slice_track[i][1],
              slice_track[object_number][0]:slice_track[object_number][1],
              slice_track[object_number+1][0]:slice_track[object_number+1][1]] = volume[count:count+S, :, :]
        count += slice_track[i][1] - slice_track[i][0]
        if direction == 'coronal':
            subject = np.swapaxes(subject, 0, 1)
        elif direction == 'axial':
            subject = np.swapaxes(subject, 0, 2)
        if (offset+i) == 23:
            ss = np.zeros(shape=(160, 192, 256))
            ss[8:152] = subject
            subject = ss
        subject = nib.Nifti1Pair(subject, np.eye(4))
        nib.save(subject, os.path.join('iSeg-2017-Testing-Results', name))

def dice_coef(predict, true):
    smooth = 1.0
    predict = predict.flatten()
    true = true.flatten()
    intersection = np.sum(predict*true)
    dice = (2*intersection + smooth)*1.0/(np.sum(predict)+np.sum(true)+smooth)
    return dice

def attach_2_original_brain1(volume, slice_track, size_track, direction, tissue, test=True):

    count = 0
    if test:
        object_number = 13  # testing data
        offset = 11
    else:
        object_number = 10  # training data
        offset = 1
    for i in range(object_number):
        name = 'subject-' + str(offset+i) + tissue + direction + '-label.img'
        print('attaching prediction volume ' + str(offset + i) + 'to original volume ( ' + name + ' )')
        subject = np.zeros(shape=(size_track[i][0], size_track[i][1], size_track[i][2]))
        S = slice_track[i][1] - slice_track[i][0]
        subject[slice_track[i][0]:slice_track[i][1],
              slice_track[object_number][0]:slice_track[object_number][1],
              slice_track[object_number+1][0]:slice_track[object_number+1][1]] = volume[count:count+S, :, :]
        count += slice_track[i][1] - slice_track[i][0]
        if direction == 'coronal':
            subject = np.swapaxes(subject, 0, 1)
        elif direction == 'axial':
            subject = np.swapaxes(subject, 0, 2)
        if (offset+i) == 23:
            ss = np.zeros(shape=(160, 192, 256))
            ss[8:152] = subject
            subject = ss
        subject = nib.Nifti1Pair(subject, np.eye(4))
        nib.save(subject, os.path.join('iSeg-2017-Testing-Results', name))


def fuse_argmax1(predict_matrix, S, axis=3):  # in shape (num, image_size, image_size, predict_class)
    predict = np.argmax(predict_matrix, axis=axis)
    seg = np.zeros(shape=predict.shape, dtype=np.uint8)
    seg[np.where(predict == 3)] = 250
    seg[np.where(predict == 2)] = 150
    seg[np.where(predict == 1)] = 50
    seg[np.where(predict == 0)] = 0
    return np.expand_dims(seg, axis=3)
