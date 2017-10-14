
import const
import h5py
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate, Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from model import make_model, symmetrize_batch
import numpy as np
import os
import replay
import simplejson as json
import yaml


np.random.seed(const.SEED)

TRAIN_SPLIT = 0.9
EPOCHS = 50
BATCH_SIZE = 512


def train_imitation(data_path, model, output_path):
    print model.summary()

    n = num_samples(data_path)
    ntrain = int(TRAIN_SPLIT * n)
    checkpoint = ModelCheckpoint(output_path,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
    model.fit_generator(gen_data(data_path, 0, ntrain),
                        validation_data=gen_data(data_path, ntrain, n),
                        steps_per_epoch=ntrain // BATCH_SIZE,
                        validation_steps=(n-ntrain) // BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpoint])


def gen_data(data_path, start, stop):
    with h5py.File(data_path, 'r') as f:
        while True:
            # m = np.random.randint(start, stop, BATCH_SIZE)
            # x = np.array([f['pred'][i] for i in m])
            # y = np.array([f['dep'][i] for i in m])
            # x = list(np.swapaxes(x, 0, 1))
            # yield x, y

            i = np.random.randint(start, stop-BATCH_SIZE)
            x = f['pred'][i:i+BATCH_SIZE]
            y = f['dep'][i:i+BATCH_SIZE]

            # before: batch, direction, y, x, feature
            # after: batch, y, x, direction, feature
            x = np.moveaxis(x, 1, 3)
            yield symmetrize_batch(x, y)


def num_samples(data_path):
    with h5py.File(data_path, 'r') as f:
        return f['pred'].shape[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.model:
        from keras.models import load_model
        model = load_model(args.model)
    else:
        model = make_model()

    train_imitation(args.data, model, args.output)

