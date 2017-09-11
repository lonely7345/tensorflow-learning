# -*- coding: utf-8 -*-
#卷积
# setup-only-ignore
import tensorflow as tf
import numpy as np

# setup-only-ignore
sess = tf.InteractiveSession()

input_batch = tf.constant([
        [  # First Input
            [[0.0], [1.0]],
            [[2.0], [3.0]]
        ],
        [  # Second Input
            [[2.0], [4.0]],
            [[6.0], [8.0]]
        ]
    ])

kernel = tf.constant([
        [
            [[2.0, 2.0]]
        ]
    ])

conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

print sess.run(conv2d)