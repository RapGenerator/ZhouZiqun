#!/usr/bin/python
# -*- coding: UTF-8 -*-
#########################################################################
# File Name: test1.4_ref.py
# Author: Bittersweet
# ###############
# Mail: zhouziqun@cool2645.com
# Created Time: 2018年07月25日 星期三 18时02分37秒
#########################################################################
import tensorflow as tf
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
EPOCHS = 1000
LEARNING_RATE = 0.1

class XOR(tf.keras.Model):
    def __init__(self):
        super(XOR, self).__init__(self)
        self.dense1 = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, input_shape=(2, ))
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.to_float(tf.equal(y_true, tf.round(y_pred))))

def loss(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)


class Optimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate):
        super(Optimizer, self).__init__()
        self._learning_rate = learning_rate

    def get_updates(self, loss, params):
        updates = []
        grads = tf.gradients(loss, params)
        for p, g in zip(params, grads):
            v = self._learning_rate * g
            op = tf.assign_sub(p, v)
            updates.append(op)
        return updates

def run():
    m = XOR()
    m.compile(loss=loss, optimizer=Optimizer(LEARNING_RATE), metrics=[accuracy])
    #m.compile(loss=tf.losses.mean_squared_error, optimizer=Optimizer(LEARNING_RATE), metrics=[accuracy])
    #m.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), metrics=[accuracy])
    m.fit(X, y, epochs=EPOCHS)
    print(m.predict(X))

if __name__ == "__main__":
    run()
