 
        ]
    ])

kernel = tf.constant([
        [
            [[2.0, 2.0]]
        ]
    ])

conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

print sess.run(conv2d)