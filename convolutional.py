from __future__ import absolute_import
import functools

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers.convolutional import Convolution3D
from keras.utils.generic_utils import get_custom_objects
from keras.utils.conv_utils import conv_output_length
#from keras.utils.conv_utils import normalize_data_format
from keras.backend.common import normalize_data_format
import numpy as np


class Deconvolution3D(Convolution3D):
    """Transposed convolution operator for filtering windows of 3-D inputs.
    The need for transposed convolutions generally arises from the desire to
    use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape
    of the output of some convolution to something that has the shape
    of its input while maintaining a connectivity pattern
    that is compatible with said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128, 128)` for a 128x128x128 volume with
    three channels.
    To pass the correct `output_shape` to this layer,
    one could use a test model to predict and observe the actual output shape.
    # Examples
    ```python
        # TH dim ordering.
        # apply a 3x3x3 transposed convolution
        # with stride 1x1x1 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 3, 14, 14, 14),
                                  padding='valid',
                                  input_shape=(3, 12, 12, 12)))
        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12, 12))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 3, 14, 14, 14)
        # apply a 3x3x3 transposed convolution
        # with stride 2x2x2 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 3, 25, 25, 25),
                                  strides=(2, 2, 2),
                                  padding='valid',
                                  input_shape=(3, 12, 12, 12)))
        model.summary()
        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12, 12))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 3, 25, 25, 25)
    ```
    ```python
        # TF dim ordering.
        # apply a 3x3x3 transposed convolution
        # with stride 1x1x1 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 14, 14, 14, 3),
                                  padding='valid',
                                  input_shape=(12, 12, 12, 3)))
        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 12, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 14, 14, 14, 3)
        # apply a 3x3x3 transposed convolution
        # with stride 2x2x2 and 3 output filters on a 12x12x12 image:
        model = Sequential()
        model.add(Deconvolution3D(3, 3, 3, 3, output_shape=(None, 25, 25, 25, 3),
                                  strides=(2, 2, 2),
                                  padding='valid',
                                  input_shape=(12, 12, 12, 3)))
        model.summary()
        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 12, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)  # (None, 25, 25, 25, 3)
    ```
    # Arguments
        filters: Number of transposed convolution filters to use.
        kernel_size: kernel_size: An integer or tuple/list of 3 integers, specifying the
            dimensions of the convolution window.
        output_shape: Output shape of the transposed convolution operation.
            tuple of integers
            `(nb_samples, filters, conv_dim1, conv_dim2, conv_dim3)`.
             It is better to use
             a dummy input and observe the actual output shape of
             a layer, as specified in the examples.
        init: name of initialization function for the weights of the layer
            (see [initializers](../initializers.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano/TensorFlow function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        padding: 'valid', 'same' or 'full'
            ('full' requires the Theano backend).
        strides: tuple of length 3. Factor by which to oversample output.
            Also called strides elsewhere.
        kernel_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        bias_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the use_bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        kernel_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        bias_constraint: instance of the [constraints](../constraints.md) module,
            applied to the use_bias.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode is it at index 4.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        use_bias: whether to include a use_bias
            (i.e. make the layer affine rather than linear).
    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(samples, filters, nekernel_conv_dim1, nekernel_conv_dim2, nekernel_conv_dim3)` if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, nekernel_conv_dim1, nekernel_conv_dim2, nekernel_conv_dim3, filters)` if data_format='channels_last'.
        `nekernel_conv_dim1`, `nekernel_conv_dim2` and `nekernel_conv_dim3` values might have changed due to padding.
    # References
        - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
        - [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
        - [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self, filters, kernel_size,
                 output_shape, activation=None, weights=None,
                 padding='valid', strides=(1, 1, 1), data_format=None,
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        if padding not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Deconvolution3D:', padding)
        if len(output_shape) == 4:
            # missing the batch size
            output_shape = (None,) + tuple(output_shape)

        self.output_shape_ = output_shape

        super(Deconvolution3D, self).__init__(kernel_size=kernel_size,
                                              filters=filters,
                                              activation=activation,
                                              weights=weights,
                                              padding=padding,
                                              strides=strides,
                                              data_format=data_format,
                                              kernel_regularizer=kernel_regularizer,
                                              bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint,
                                              use_bias=use_bias,
                                              kernel_initializer=kernel_initializer,
                                              bias_initializer=bias_initializer,
                                              **kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            conv_dim1 = self.output_shape_[2]
            conv_dim2 = self.output_shape_[3]
            conv_dim3 = self.output_shape_[4]
            return (input_shape[0], self.filters, conv_dim1, conv_dim2, conv_dim3)
        elif self.data_format == 'channels_last':
            conv_dim1 = self.output_shape_[1]
            conv_dim2 = self.output_shape_[2]
            conv_dim3 = self.output_shape_[3]
            return (input_shape[0], conv_dim1, conv_dim2, conv_dim3, self.filters)
        else:
            raise ValueError('Invalid data format: ', self.data_format)

    def call(self, x, mask=None):
        kernel_shape = K.get_value(self.kernel).shape
        output = K.deconv3d(x, self.kernel, self.output_shape_,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                            filter_shape=kernel_shape)
        if self.use_bias:
            if self.data_format == 'channels_first':
                output += K.reshape(self.bias, (1, self.filters, 1, 1, 1))
            elif self.data_format == 'channels_last':
                output += K.reshape(self.bias, (1, 1, 1, 1, self.filters))
            else:
                raise ValueError('Invalid data_format: ', self.data_format)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_shape': self.output_shape_}
        base_config = super(Deconvolution3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


Deconv3D = Deconvolution3D
get_custom_objects().update({'Deconvolution3D': Deconvolution3D})
get_custom_objects().update({'Deconv3D': Deconv3D})