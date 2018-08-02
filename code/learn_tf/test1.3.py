#!/usr/bin/python
# -*- coding: UTF-8 -*-
#########################################################################
# File Name: test1.3.py
# Author: Bittersweet
# ###############
# Mail: zhouziqun@cool2645.com
# Created Time: 2018年07月25日 星期三 17时08分09秒
#########################################################################
import tensorflow as tf


EPOCHS = 100
LEARNING_RATE = 0.1


def find_root(f, x_init, y_expected, learning_rate):
    x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(x_init))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(EPOCHS):
            y_predict = f(x)
            loss = tf.abs(y_predict - y_expected)
            y_grad = tf.gradients(loss, [x])
            op = tf.assign_sub(x, learning_rate * y_grad[0])
            op = tf.Print(op, [x], "updates: ")
            sess.run(op)
        return sess.run(x)

#f = lambda x: 3 * x
#x_init = 1
#y_expected = 5
if __name__ == "__main__":
    find_root(lambda x: 3*x, 1, 5, LEARNING_RATE)
