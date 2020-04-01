from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from keras import datasets, layers, models, losses
from keras.callbacks import TensorBoard
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def creating():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32,32,3)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
    print(model.summary())
    return model

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    model = creating()

    log_dir = 'logs/{}'.format(datetime.datetime.now())
    tensor_board = TensorBoard(log_dir = log_dir, histogram_freq = 1)
    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels), callbacks = [tensor_board])
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_loss)
    print(test_acc)

main()

