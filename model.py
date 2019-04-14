import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Reshape, Conv3D, MaxPooling3D, UpSampling3D, Activation, concatenate, BatchNormalization
from keras import optimizers
from config import *
from convolutional import Deconv3D

def get_3DUnet(num_input_channels):
    inputs = Input((image_size, image_size, image_size, num_input_channels))
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(inputs)
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(pool1)
    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(pool2)
    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(pool3)
    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(pool4)
    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
    conv6 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(up6)
    conv6 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
    conv7 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(up7)
    conv7 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
    conv8 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(up8)
    conv8 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
    conv9 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(up9)
    conv9 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv9)

    conv10 = Conv3D(3, (1, 1, 1), padding="same", activation='relu')(conv9)

    conv10 = Reshape((image_size, image_size, image_size, 3))(conv10)
    conv11 = Activation('softmax')(conv10)

    model = Model(outputs=conv11, inputs=inputs)

    # sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
'''
def get_3DUnet_Deeper(num_input_channels):
    inputs = Input((image_size, image_size, image_size, num_input_channels))
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(inputs)
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(pool1)
    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(pool2)
    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(pool3)
    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(pool4)
    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
    conv6 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(up6)
    conv6 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv6)
    soft1 = Deconv3D(3, (3, 3, 3), output_shape=(16, 16, 16, 3), strides=(8, 8, 8), padding="valid", input_shape=(2, 2, 2, filter_size*16))(conv6)
    soft1 = Activation('softmax')(soft1)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
    conv7 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(up7)
    conv7 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv7)
    soft2 = Deconv3D(3, (3, 3, 3), strides=(4, 4, 4), padding="valid", input_shape=(4, 4, 4, filter_size*8))(conv7)
    soft2 = Activation('softmax')(soft2)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
    conv8 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(up8)
    conv8 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv8)
    soft3 = Deconv3D(3, (3, 3, 3), output_shape=(16, 16, 16, 3), strides=(2, 2, 2), padding="valid", input_shape=(8, 8, 8, filter_size*3))(conv8)
    soft3 = Activation('softmax')(soft3)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
    conv9 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(up9)
    conv9 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv9)


    conv10 = Conv3D(3, (1, 1, 1), padding="same", activation='relu')(conv9)
    conv10 = Reshape((image_size, image_size, image_size, 3))(conv10)
    conv10 = Activation('softmax')(conv10)

    DEEP = conv10 + soft1 + soft2 + soft3

    model = Model(outputs=DEEP, inputs=inputs)

    # sgd = optimizers.SGD(lr=0.01, decay=4e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
get_3DUnet_Deeper(2)
'''
'''
def get_3DUnet(num_input_channels):
    inputs = Input((image_size, image_size, image_size, num_input_channels))
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = BatchNormalization()(pool4)
    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(filter_size*16, (3, 3, 3), padding="same", activation='relu')(conv5)


    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=4)
    conv6 = BatchNormalization()(up6)
    conv6 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(filter_size*8, (3, 3, 3), padding="same", activation='relu')(conv6)


    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=4)
    conv7 = BatchNormalization()(up7)
    conv7 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(filter_size*4, (3, 3, 3), padding="same", activation='relu')(conv7)


    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=4)
    conv8 = BatchNormalization()(up8)
    conv8 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(filter_size*2, (3, 3, 3), padding="same", activation='relu')(conv8)


    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=4)
    conv9 = BatchNormalization()(up9)
    conv9 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv3D(filter_size, (3, 3, 3), padding="same", activation='relu')(conv9)

    conv10 = BatchNormalization()(conv9)
    conv10 = Conv3D(3, (1, 1, 1), padding="same", activation='relu')(conv10)


    conv10 = core.Reshape((image_size, image_size, image_size, 3))(conv10)
    conv11 = core.Activation('softmax')(conv10)

    model = Model(outputs=conv11, inputs=inputs)


    sgd = optimizers.SGD(lr=0.0001, decay=4e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss='mse')
    model.summary()
    print 'Using Batch_normal, RELU, CONV3D'
    return model


'''