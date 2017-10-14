
import const
from keras.layers import (
    Activation, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, Lambda,
    Reshape)
from keras.models import Model
import numpy as np


NUM_FEATURES = 147
NUM_LAYERS = 12
NUM_FILTERS = 32
KERNEL_SIZE = 3


def make_model():
    layers = []
    for _ in range(NUM_LAYERS):
        layer = Conv2D(NUM_FILTERS,
                       (KERNEL_SIZE, KERNEL_SIZE),
                       padding='same',
                       activation='relu')
        layers.append(layer)
    layers.append(Conv2D(1, (1, 1), padding='same'))

    branches = []
    inputs = []
    shape = const.BOARD_SIZE, const.BOARD_SIZE, NUM_FEATURES
    for i in range(const.NUM_DIRECTIONS):
        x = Input(shape=shape)
        inputs.append(x)
        for layer in layers:
            x = layer(x)
        branches.append(x)

    x = Concatenate()(branches)
    x = Flatten()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def symmetrize_batch(x, y):
    y = y.reshape((-1, const.BOARD_SIZE, const.BOARD_SIZE, const.NUM_DIRECTIONS))

    def random_slice(axis):
        x_index = [slice(None)] * x.ndim
        y_index = [slice(None)] * y.ndim
        n = x.shape[axis]
        x_index[axis] = y_index[axis] = np.random.choice(np.arange(n), n/2, replace=False)
        return x_index, y_index

    # randomize along batch and move direction axes
    for axis in [0, 3]:
        m1, m2 = random_slice(axis)
        x[m1] = np.rot90(x[m1], axes=(1, 2))
        y[m2] = np.rot90(y[m2], axes=(1, 2))

        for flip_axis in [1, 2]:
            m = random_slice(axis)
            x[m1] = np.flip(x[m1], axis=flip_axis)
            y[m2] = np.flip(y[m2], axis=flip_axis)

    y = y.reshape((-1, const.NUM_DIRECTIONS * const.BOARD_SIZE ** 2))
    return x, y

    # m, _, _, n = x.shape

    # r = np.random.choice(np.arange(m), m/2, replace=False)
    # x[r, :, :, :] = np.rot90(x[r, :, :, :], axes=(1, 2))
    # y[r, :, :, :] = np.rot90(y[r, :, :, :], axes=(1, 2))

    # for axis in [1, 2]:
    #     r = np.random.choice(np.arange(n), n/2, replace=False)
    #     x[r, :, :, :] = np.flip(x[r, :, :, :], axis=axis)
    #     y[r, :, :, :] = np.flip(y[r, :, :, :], axis=axis)

    # r = np.random.choice(np.arange(m), m/2, replace=False)
    # x[:, :, :, r] = np.rot90(x[:, :, :, r], axes=(1, 2))
    # y[:, :, :, r] = np.rot90(y[:, :, :, r], axes=(1, 2))

    # for axis in [1, 2]:
    #     r = np.random.choice(np.arange(n), n/2, replace=False)
    #     x[:, :, :, r] = np.flip(x[:, :, :, r], axis=axis)
    #     y[:, :, :, r] = np.flip(y[:, :, :, r], axis=axis)
    
    # m, _, _, n = x.shape
    # for i in range(m):
    #     for j in range(n):
    #         k = np.random.randint(4)
    #         x[i, :, :, j] = np.rot90(x[i, :, :, j], k=k)
    #         y[i, :, :, j] = np.rot90(y[i, :, :, j], k=k)
    #         if np.random.randint(2)
    #             x[i, :, :, j] = np.flipud(x[i, :, :, j])
    #             y[i, :, :, j] = np.flipud(y[i, :, :, j])
    return x, y

