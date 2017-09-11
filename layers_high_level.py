# -*- coding: utf-8 -*-
#高级层

import tensorflow as tf



image_input = tf.constant([
            [
                [[0., 0., 0.], [255., 255., 255.], [254., 0., 0.]],
                [[0., 191., 0.], [3., 108., 233.], [0., 191., 0.]],
                [[254., 0., 0.], [255., 255., 255.], [0., 0., 0.]]
            ]
        ])

conv2d = tf.contrib.layers.convolution2d(
    image_input,
    num_outputs=4,
    kernel_size=(1,1),          # It's only the filter height and width.
    activation_fn=tf.nn.relu,
    stride=(1, 1),              # Skips the stride values for image_batch and input_channels.
    trainable=True)

sess = tf.InteractiveSession()

# It's required to initialize the variables used in convolution2d's setup.
sess.run(tf.initialize_all_variables())

print sess.run(conv2d)