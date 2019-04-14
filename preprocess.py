import sys
import os
import nibabel as nib
import numpy as np
import skimage.exposure as skiex
from keras.utils.np_utils import to_categorical
from config import *
from function import normallize
from random import randint
import matplotlib.pyplot as plt


def sort_trim(volume, half):
    volume = np.swapaxes(volume, 0, 1)
    volume = trim_zeroSlice(volume, half)
    volume = np.swapaxes(volume, 0, 1)
    volume = np.swapaxes(volume, 0, 2)
    volume = trim_zeroSlice(volume, half)
    volume = np.swapaxes(volume, 0, 2)
    return volume

# get the directory of file and sort them in order (T1, T2, and label)
def trim_zeroSlice(volume, half=True):
    reshape = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
    trim_index = np.sum(reshape, axis=1)
    volume_trim = volume[np.min(np.nonzero(trim_index)): np.max(np.nonzero(trim_index)) + 1]
    if np.mod(volume_trim.shape[0], image_size) == 0:
        extra_pad = image_size
    else:
        extra_pad = image_size - np.mod(volume_trim.shape[0], image_size)
    volume_pad = np.zeros(shape=(volume_trim.shape[0]+extra_pad, volume_trim.shape[1],
                                 volume_trim.shape[2]), dtype=np.float32)
    if half:
        volume_pad[int(extra_pad / 2): int(extra_pad / 2) + volume_trim.shape[0], :, :] = volume_trim
    else:
        volume_pad[:volume_trim.shape[0], :, :] = volume_trim
    return volume_pad

def CLAHE(T_volume, label=False):
    T_mask = np.zeros((T_volume.shape), dtype=np.float32)
    T_mask[np.where(T_volume > 0.0)] = 1
    T = np.reshape(T_volume, (T_volume.shape[0], T_volume.shape[1] * T_volume.shape[2]))
    if label==False:
        T = skiex.equalize_adapthist(T.astype(np.uint16), clip_limit=0.2)
    T = np.reshape(T, (T_volume.shape[0], T_volume.shape[1], T_volume.shape[2])).astype(np.float32)
    return T * T_mask, T_mask

def read_volume(volume_directory, label, half):
    volume = nib.load(volume_directory).get_data().astype(np.float32)  # convert to float32 from int16
    volume = np.reshape(volume, (volume.shape[0], volume.shape[1], volume.shape[2]))
    volume, T_mask = CLAHE(volume, label=label)
    volume = trim_zeroSlice(volume, half)
    T_mask = trim_zeroSlice(T_mask, half)
    return volume, T_mask

def append_volume(volume_directory, half, label):
    volume, T_mask = read_volume(volume_directory[0], label, half)
    for i in range(len(volume_directory) - 1):
        new_volume, T_mask = read_volume(volume_directory[i + 1], label, half)
        volume = np.append(volume, new_volume, axis=0)
    volume = sort_trim(volume, half)

    return volume

def extract_patch(volume, subsize):
    sub_volume = np.ndarray(shape=(int(volume.shape[1] / subsize * volume.shape[2] / subsize),
                                   volume.shape[0], subsize, subsize), dtype=np.float32)

    for row in range(int(volume.shape[1] / subsize)):
        for col in range(int(volume.shape[2] / subsize)):
            A = volume[:, row * subsize: (row + 1) * subsize, col * subsize: (col + 1) * subsize]
            sub_volume[int(row * volume.shape[2] / subsize + col)] = A
    return sub_volume.reshape((int(sub_volume.shape[0]*sub_volume.shape[1]/image_size),
                              image_size, image_size, image_size, 1))

def remove_zeroVolume(volume):
    S = volume.shape
    SUM = np.sum(volume.reshape(S[0], S[1] * S[2] * S[3]), 1)
    index = np.where(SUM == 0)
    return np.delete(volume, list(index), 0), index

def list_file_name(train):
    T1_list = []
    T2_list = []
    label_list = []
    Path = ''
    if train=='train':
        Path = Path_train
    elif train== 'test':
        Path = Path_test

    for dirName, subdirList, fileList in os.walk(Path):
        if subdirList == []:
            for name in fileList:
                name = name.replace("._", "")
                if 'T1.img' in name:
                    T1_list.append(Path + '/' + name)
                elif 'T2.img' in name:
                    T2_list.append(Path + '/' + name)
                elif 'label.img' in name:
                    label_list.append(Path + '/' + name)
    T1_list.sort()
    T2_list.sort()
    label_list.sort()
    return T1_list, T2_list, label_list

def encode_numClass(label_volume):
    label = np.zeros(shape=label_volume.shape, dtype=np.float32)
    label[np.where(label_volume == 10.0)] = 0.0
    label[np.where(label_volume == 150)] = 1.0
    label[np.where(label_volume == 250)] = 2.0
    return label

def suffle(dataset, labelset):  # need to suffle data and label simultaneously
    data = np.ndarray(shape=dataset.shape, dtype=np.float32)
    label = np.ndarray(shape=labelset.shape, dtype=np.float32)
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    for i in range(len(dataset)):
        data[i] = dataset[index[i]]
        label[i] = labelset[index[i]]
    return data, label

def preprocess_train_data(T1_list, T2_list, label_list, half):
    T1 = append_volume(T1_list, half, label=False)
    T2 = append_volume(T2_list, half, label=False)
    label = append_volume(label_list, half, label=True)

    T1 = extract_patch(T1, image_size)
    T2 = extract_patch(T2, image_size)
    label = extract_patch(label, image_size)

    #T1, idex = remove_zeroVolume(T1)
    #T2, idex = remove_zeroVolume(T2)
    #label, idex = remove_zeroVolume(label)
    S = label.shape

    label = encode_numClass(label)
    label = to_categorical(label, 3)
    label = np.reshape(label, (S[0], S[1], S[2], S[3], 3))
    T = np.concatenate((T1, T2), axis=4)
    del T1, T2
    return T, label



if __name__ == '__main__':
    T1_list, T2_list, label_list = list_file_name(train='train')
    T_train, label_train = preprocess_train_data(T1_list, T2_list, label_list, True)
    #T_train_half, label_train_half = preprocess_train_data(T1_list, T2_list, label_list, False)
    #T_train_half = np.rot90(T_train_half, k=1, axes=(1, 3))
    #label_train_half = np.rot90(label_train_half, k=1, axes=(1, 3))
    #T_train = np.concatenate((T_train, T_train_half), axis=0)
    #del T_train_half
    #label_train = np.concatenate((label_train, label_train_half), axis=0)
    #del label_train_half
    T_train = np.concatenate((T_train, np.rot90(T_train, k=1, axes=(2, 3))), axis=0)
    label_train = np.concatenate((label_train, np.rot90(label_train, k=1, axes=(2, 3))), axis=0)

    T_train, label_train = suffle(T_train, label_train)
    print('Mean T_train / Mean label ', np.mean(T_train), np.mean(label_train))
    print('Std T_train, / Std label', np.std(T_train), np.std(label_train))
    T_train = normallize(T_train)
    #print np.min(T_train), np.max(T_train), np.mean(T_train), np.std(T_train)
    np.savez('train' + str(image_size) + '.npy', T_train=T_train, label_train=label_train)
    print('Min T_train  / Min label', np.min(T_train), np.min(label_train))
    print('Max T_train  / Min label', np.max(T_train), np.max(label_train))
    print('Mean T_train / Mean label ', np.mean(T_train), np.mean(label_train))
    print('Std T_train, / Std label', np.std(T_train), np.std(label_train))
    print('Shape T_train/ Std label', T_train.shape, label_train.shape)



'''
T1_list, T2_list, label_list = list_file_name(train='test')
T_train, label_train = preprocess_train_data(T1_list, T2_list, label_list, True)
T_train_half, label_train_half = preprocess_train_data(T1_list, T2_list, label_list, False)
T_train_half = np.rot90(T_train_half, k=1, axes=(1, 3))
label_train_half = np.rot90(label_train_half, k=1, axes=(1, 3))
T_train = np.concatenate((T_train, T_train_half), axis=0)
del T_train_half
label_train = np.concatenate((label_train, label_train_half), axis=0)
del label_train_half
T_train = np.concatenate((T_train, np.rot90(T_train, k=1, axes=(2, 3))), axis=0)
label_train = np.concatenate((label_train, np.rot90(label_train, k=1, axes=(2, 3))), axis=0)

T_train, label_train = suffle(T_train, label_train)
print np.min(T_train), np.max(T_train), np.mean(T_train), np.std(T_train)
np.savez('train' + str(image_size) + '.npy', T_train=T_train, label_train=label_train)

print T_train.shape, label_train.shape
'''



'''
num = randint(0, len(T_train[0]) - 1)
for m in range(100):
    num +=100
    print num

    T1 = T_train[num, :, :, :, 0]
    T2 = T_train[num, :, :, :, 1]
    L = label_train[num, :, :, :, 0]
    figure, ax = plt.subplots(4, 12, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    for i in range(4):
        for j in range(4):
            image1 = T1[i * 4 + j, :, :]
            image2 = T2[i * 4 + j, :, :]
            l = L[i * 4 + j, :, :]
            ax[i, j].imshow(image1, cmap='gray')
            ax[i, j + 4].imshow(image2, cmap='gray')
            ax[i, j + 8].imshow(l, cmap='RdYlGn')
    plt.show()
'''