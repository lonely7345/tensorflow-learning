# -*- coding: utf-8 -*-
#池化层
import tensorflow as tf

# Usually the input would be output from a previous layer and not an image directly.
batch_size=1
input_height = 3
input_width = 3
input_channels = 1

layer_input = tf.constant([
        [
            [[1.0], [0.2], [1.5]],
            [[0.1], [1.2], [1.4]],
            [[1.1], [0.4], [0.4]]
        ]
    ])

# The strides will look at the entire input by using the image_height and image_width
kernel = [batch_size, input_height, input_width, input_channels]
max_pool = tf.nn.max_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")

sess = tf.InteractiveSession()

print  sess.run(max_pool)

avg_pool = tf.nn.avg_pool(layer_input, kernel, [1, 1, 1, 1], "VALID")


print  sess.run(avg_pool)