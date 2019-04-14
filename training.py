import sys
import numpy as np
#from sklearn.cross_validation import KFold
import model
import function as fn
from keras.callbacks import ModelCheckpoint  # EarlyStopping
from keras import backend as K
import pickle
from config import *

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def fit(T_train, label_1hot_encode, num_channel):
    M = model.get_3DUnet(int(num_channel))
    model_save_best = ModelCheckpoint('UNET_' + str(image_size) + '_save_best', monitor='val_loss',
                                      verbose=1, save_best_only=True)
    hist = M.fit(T_train, label_1hot_encode, batch_size=batch_size, epochs=epoch, verbose=1,
                     validation_split=0.2, shuffle=True, callbacks=[model_save_best])
    print('saving history')
    with open('./history.pickle', 'wb') as file_name:
        pickle.dump(hist.history, file_name)

if __name__ == '__main__':
#    #with np.load(sys.argv[1] + str(image_size) + '.npy.npz') as data:
    with np.load('train16.npy.npz') as data:
        #'train16' '.npy.npz'
        T_train = data['T_train']
        label_train = data['label_train']
    print(T_train.shape, label_train.shape)
    validation_ratio = 0.2
    #class_weights = fn.class_weighting(label_train, validation_ratio)
    #print(class_weights)
    T_train = fn.normallize(T_train)
    fit(T_train, label_train, 2)
#fit(T_train, label_train, class_weights, sys.argv[1], sys.argv[2])
