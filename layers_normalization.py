# -*- coding: utf-8 -*-

import tensorflow as tf


layer_input = tf.constant([
        [[[ 1.]], [[ 2.]], [[ 3.]]]
    ])

lrn = tf.nn.local_response_normalization(layer_input)
print sess.run([layer_input, lrn])

