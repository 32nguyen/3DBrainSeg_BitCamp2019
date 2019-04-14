import numpy as np
import function as fn
from preprocess import list_file_name, read_volume, extract_patch, sort_trim, remove_zeroVolume
from config import *
import model
import nibabel as nib
import matplotlib.pyplot as plt
import os

def test(T1_list, T2_list, half):
    T1, T_mask = read_volume(T1_list, label=False, half=False)
    T1 = sort_trim(T1, half)

    T2, T_mask = read_volume(T2_list, label=False, half=False)
    T2 = sort_trim(T2, half)

    T_mask = sort_trim(T_mask, half)

    return T2, T_mask

def process_test_data(T1_list, T2_list, half):
    T1, T_mask = read_volume(T1_list, label=False, half=False)
    T1 = sort_trim(T1, half)

    T2, T_mask = read_volume(T2_list, label=False, half=False)
    T2 = sort_trim(T2, half)

    T_mask = sort_trim(T_mask, half)

    T1 = extract_patch(T1, image_size)
    T2 = extract_patch(T2, image_size)

    shape_withZero = T1.shape
    T1, _ = remove_zeroVolume(T1)
    T2, idex = remove_zeroVolume(T2)

    T = np.concatenate((T1, T2), axis=4)
    return T, T_mask, shape_withZero, idex

def insert_zeroSlice(volume, o_shape, zero_index):
    index = zero_index[0]
    matrix = np.ones(shape=(o_shape[0], o_shape[1], o_shape[2], o_shape[3], 3), dtype=np.float32)
    for i in index:
        matrix[i] = 0
    count = 0
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i]) != 0:
            matrix[i] = volume[count]
            count +=1
    return matrix

def fuse_argmax(predict_matrix, axis=4):  # in shape (num, image_size, image_size, image_size, predict_class)
    predict = np.argmax(predict_matrix, axis=axis)
    seg = np.zeros(shape=predict.shape, dtype=np.uint8)
    seg[np.where(predict == 2)] = 250
    seg[np.where(predict == 1)] = 150
    seg[np.where(predict == 0)] = 50
    return seg

def reshape_brain(predict, mask, subsize):
    brain = np.ndarray(shape=(mask.shape), dtype=np.float32)
    predict = np.reshape(predict, ((mask.shape[1] / subsize * mask.shape[2] / subsize,
                    mask.shape[0], subsize, subsize)))
    for row in range(brain.shape[1] / subsize):
        for col in range(brain.shape[2] / subsize):
            brain[:, row * subsize: (row + 1) * subsize,
            col * subsize: (col + 1) * subsize] = predict[row * brain.shape[2] / subsize + col]
    return brain

def brain_attach(raw_T2, predict):
    predict.astype(np.uint8)
    brain = np.zeros(shape=raw_T2.shape, dtype=np.uint8)
    volume = np.reshape(raw_T2, (raw_T2.shape[0], raw_T2.shape[1] * raw_T2.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind0_min_raw = np.min(np.nonzero(trim_index))
    ind0_max_raw = np.max(np.nonzero(trim_index))

    volume = np.swapaxes(raw_T2, 0, 1)
    volume = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind1_min_raw = np.min(np.nonzero(trim_index))
    ind1_max_raw = np.max(np.nonzero(trim_index))

    volume = np.swapaxes(raw_T2, 0, 2)
    volume = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind2_min_raw = np.min(np.nonzero(trim_index))
    ind2_max_raw = np.max(np.nonzero(trim_index))

    volume = np.reshape(predict, (predict.shape[0], predict.shape[1] * predict.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind0_min_mask = np.min(np.nonzero(trim_index))
    ind0_max_mask = np.max(np.nonzero(trim_index))

    volume = np.swapaxes(predict, 0, 1)
    volume = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind1_min_mask = np.min(np.nonzero(trim_index))
    ind1_max_mask = np.max(np.nonzero(trim_index))

    volume = np.swapaxes(predict, 0, 2)
    volume = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
    trim_index = np.sum(volume, axis=1)
    ind2_min_mask = np.min(np.nonzero(trim_index))
    ind2_max_mask = np.max(np.nonzero(trim_index))
    #print(brain.shape, T_mask.shape)
    brain[ind0_min_raw : ind0_max_raw + 1, ind1_min_raw : ind1_max_raw + 1, ind2_min_raw : ind2_max_raw + 1] = \
        predict[ind0_min_mask : ind0_max_mask + 1, ind1_min_mask : ind1_max_mask + 1, ind2_min_mask : ind2_max_mask + 1]
    return brain

T1_list, T2_list, label_list = list_file_name(train="test")  # label_list = []
# get hearder from training image
label = nib.load('./iSeg-2017-Training/subject-1-label.img')
head = label.header
#print(head)

M = model.get_3DUnet(num_input_channels=2)
print("Loading model")
M.load_weights('UNET_16_save_best')

for i in range(0, 1):
    #T_test, T_mask = test(T1_list[i], T2_list[i], False)
    #print T_test.shape, T_mask.shape

    T_test, T_mask, shape_withZero, zero_index = process_test_data(T1_list[i], T2_list[i], False)
    T_test = fn.normallize(T_test)
    probability = M.predict(T_test, batch_size=batch_size, verbose=2)
    del T_test
    probability = insert_zeroSlice(probability, shape_withZero, zero_index)
    probability = fuse_argmax(probability)
    probability = reshape_brain(probability, T_mask, image_size) * T_mask
    raw_T2 = nib.load(T2_list[i]).get_data().astype(np.float32)  # convert to float32 from int16
    raw_T2 = np.reshape(raw_T2, (raw_T2.shape[0], raw_T2.shape[1], raw_T2.shape[2]))

    raw_T1 = nib.load(T1_list[i]).get_data().astype(np.float32)  # convert to float32 from int16
    raw_T1.reshape(raw_T1.shape[0], raw_T1.shape[1], raw_T1.shape[2])

    brain = brain_attach(raw_T2, probability)
    brain = np.expand_dims(brain, axis=4)

    name = 'result.img'
    print('save ressult as:', name)
    subject = nib.AnalyzeImage(brain, np.eye(4), header=head)
    #if not os.path.exists('./iSeg-2017-Testing-Results'):
    #    os.makedirs('./iSeg-2017-Testing-Results')
    nib.save(subject, name)
    #np.save('save result as result.npy', brain)


