from keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
import tensorflow as tf
import numpy as np


"""Loading the dataset"""
def load_data(dataset):
    (train_X, train_y_1), (test_X, test_y_1) = dataset.load_data()
    n_class_1=10

    train_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in train_y_1)
    test_y_2 = list(0 if y in [5,7,9] else 1 if y in [3,6,8] else 2 for y in test_y_1)
    n_class_2 = 3

    train_X = np.expand_dims(train_X, axis=3)
    train_X = (train_X / 127.5) -1 # squishing range of x values from [0,255] to [-1,1]
    test_X = np.expand_dims(test_X, axis=3)
    test_X = (test_X / 127.5) -1

    train_y_1 = to_categorical(train_y_1, n_class_1)
    test_y_1 = to_categorical(test_y_1, n_class_1)
    train_y_2 = to_categorical(train_y_2, n_class_2)
    test_y_2 = to_categorical(test_y_2, n_class_2)
    return train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2

"""CNN model architecture for the fashion dataset. For comparison to MTL model."""
def CNN_fashion(n):
    model = tf.keras.Sequential([
    # 3 conv layers, with max pool layers after first 2
    Conv2D(32, kernel_size=3, strides = 1, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides = 1, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, strides = 1, activation='relu'),
    Flatten(),

    # dense layers
    Dense(3136, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(100, activation='relu'), # extra dense layer
    Dense(n, activation = 'softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model


def custom_loss(losses, l):
    loss = (l * losses['item']) + ((1-l) * losses['category'])
    return loss


"""MTL structure learnt from Coursera - Creating Multi Task Models With Keras [Online],
n.d. Available from: https://rhyme.com/run/SLU8RAZTJE22U0LBWDT4 [Accessed 27 June 2022]."""
def mtl():
    # hard shared layers
    input_ = Input(shape=(28,28,1), name='input')
    conv_1 = Conv2D(32, kernel_size=3, strides = 1, activation='relu', name='conv_1')(input_)
    pool_1 = MaxPool2D(pool_size=2, strides=2, name='pool_1')(conv_1)
    conv_2 = Conv2D(64, kernel_size=3, strides = 1, activation='relu', name='conv_2')(pool_1)
    pool_2 = MaxPool2D(pool_size=2, strides=2, name='pool_2')(conv_2)
    conv_3 = Conv2D(128, kernel_size=3, strides = 1, activation='relu', name='conv_3')(pool_2)
    flat_1 = Flatten(name='flat_1')(conv_3)
    dense_1 = Dense(3136, activation='relu', name='dense_1')(flat_1)

    # now the model diverges for the 2 tasks
    t1_dense_2 = Dense(1024, activation='relu', name = 't1_dense_2')(dense_1)
    t1_dense_3 = Dense(100, activation='relu', name = 't1_dense_3')(t1_dense_2)
    item = Dense(10, activation='softmax', name = 'item')(t1_dense_3)

    t2_dense_2 = Dense(1024, activation='relu', name = 't2_dense_2')(dense_1)
    t2_dense_3 = Dense(100, activation='relu', name = 't2_dense_3')(t2_dense_2)
    category = Dense(3, activation='softmax', name = 'category')(t2_dense_3)

    model = tf.keras.models.Model(input_, [item, category])

    return model
