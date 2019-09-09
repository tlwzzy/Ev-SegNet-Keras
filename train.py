import sys

from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2, l1
import numpy as np
import threading
import os
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from networks import unet

from keras.models import Model, model_from_json
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, merge
from keras.layers.core import Lambda
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import cv2
import time

from networks.unet import IoU_fun, mean_iou

# seed 1234 is used for reproducibility
np.random.seed(seed=1234)
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

## 0:Cross View, 1:Cross Subject
subject = 1

## Data root
data_root = '/root/Ev-SegNet-old/datas/lanes'

out_dir_name = 'MANs_subject'

## Parameters
loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9

activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
# optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
reg = l2(1.e-4)

batch_size = 16
epochs = 100
n_classes = 5
scale_w = 352
scale_h = 224
channel = 1

samples_per_epoch = 3589
samples_per_validation = 1835


class data_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def data_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return data_iter(f(*a, **kw))

    return g


@data_generator
def train_data():
    image_root_folder = os.path.join(data_root, "images")
    event_root_folder = os.path.join(data_root, "events")
    label_root_folder = os.path.join(data_root, "labels")

    train_x_files = os.path.join(image_root_folder, 'train')
    train_y_files = os.path.join(label_root_folder, 'train')

    train_x_list = os.listdir(train_x_files)
    train_y_list = os.listdir(train_y_files)

    X = np.zeros((batch_size, scale_h, scale_w, channel))
    Y = np.zeros((batch_size, scale_h, scale_w, n_classes))
    # print("++++++++++++++++++++++", Y.shape)
    batch_count = 0
    temp = 1
    while True:
        indices = list(range(0, samples_per_epoch))
        np.random.shuffle(indices)

        for index in indices:
            # print(os.path.join(train_x_files, train_x_list[index]))
            value = cv2.imread(os.path.join(train_x_files, train_x_list[index]), 0)
            label = cv2.imread(os.path.join(train_y_files, train_y_list[index]), 0)

            x = cv2.resize(value, (scale_h, scale_w))
            label = cv2.resize(label, (scale_h, scale_w))
            label = np.reshape(label, [scale_h*scale_w, 1])
            label = to_categorical(label, num_classes=n_classes)
            label = np.reshape(label, [scale_h, scale_w, n_classes])
            x.setflags(write=1)

            X[batch_count] = np.reshape(x, [scale_h, scale_w, channel])
            Y[batch_count] = label
            batch_count += 1

            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, scale_h, scale_w, channel))
                Y = np.zeros((batch_size, scale_h, scale_w, n_classes))
                temp += 1
                batch_count = 0
                yield (ret_x, ret_y)


@data_generator
def test_data():
    image_root_folder = os.path.join(data_root, "images")
    event_root_folder = os.path.join(data_root, "events")
    label_root_folder = os.path.join(data_root, "labels")

    test_x_files = os.path.join(image_root_folder, 'test')
    test_y_files = os.path.join(label_root_folder, 'test')

    test_x_list = os.listdir(test_x_files)
    test_y_list = os.listdir(test_y_files)

    X = np.zeros((batch_size, scale_h, scale_w, channel))
    Y = np.zeros((batch_size, scale_h, scale_w, n_classes))
    batch_count = 0
    temp = 1
    while True:
        indices = list(range(0, samples_per_validation))
        np.random.shuffle(indices)

        for index in indices:
            value = cv2.imread(os.path.join(test_x_files, test_x_list[index]), 0)
            label = cv2.imread(os.path.join(test_y_files, test_y_list[index]), 0)

            x = cv2.resize(value, (scale_h, scale_w))
            label = cv2.resize(label, (scale_h, scale_w))
            label = np.reshape(label, [scale_h * scale_w, 1])
            label = to_categorical(label, num_classes=n_classes)
            label = np.reshape(label, [scale_h, scale_w, n_classes])
            x.setflags(write=1)

            X[batch_count] = x.reshape((scale_h, scale_w, channel))
            Y[batch_count] = label
            batch_count += 1

            if batch_count == batch_size:
                ret_x = X
                ret_y = Y
                X = np.zeros((batch_size, scale_h, scale_w, channel))
                Y = np.zeros((batch_size, scale_h, scale_w, n_classes))
                batch_count = 0
                temp += 1
                yield (ret_x, ret_y)


def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice no.
    [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]


def to_multi_gpu(model, n_gpus=4):
    """Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.
    Each GPU gets a slice of the input batch, applies the model on that
    slice
    and later the outputs of the models are concatenated to a single
    tensor,
    hence the user sees a model that behaves the same as the original.
    """

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.input_names[0])
    towers = []
    device = [0, 1, 2, 3]
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(device[g])):
            slice_g = Lambda(slice_batch, lambda shape: shape,
                             arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(inputs=[x], outputs=merged)


def train():
    model = unet.unet(input_size=(scale_h, scale_w, channel), num_class=n_classes)

    # model = to_multi_gpu(model, 4)
    # model.load_weights('039_0.827.hdf5')
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy', mean_iou])
    
    if not os.path.exists('weights/' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + out_dir_name):
        os.makedirs('weights/' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + out_dir_name)
    weight_path = 'weights/' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + out_dir_name + '/{epoch:03d}_{val_mean_iou:0.3f}.hdf5'

    # serialize weight to h5
    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_mean_iou',
                                 verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0001)
    csv_logger = CSVLogger(
        "logs/" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) + 'log_MANs_9_subject.csv',
        append=True, separator=';')
    callbacks_list = [checkpoint, reduce_lr, csv_logger]

    model.fit_generator(train_data(),
                        steps_per_epoch=samples_per_epoch / batch_size + 1,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=test_data(),
                        validation_steps=samples_per_validation / batch_size + 1,
                        workers=1,
                        initial_epoch=0
                        )


if __name__ == "__main__":
    train()
