# -*- coding: utf-8 -*-

import tensorflow as tf

sess = tf.Session()


image_filename = "./data/IMG_0938.jpg"

filename_queue  =  tf.train.string_input_producer(
         [image_filename]
         )

image_reader = tf.WholeFileReader()

_,image_file =  image_reader.read(filename_queue)

image = tf.image.decode_jpeg(image_file)

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess,coord=coord)

print sess.run(image)

filename_queue.close(cancel_pending_enqueues=True)

coord.request_stop()
coord.join(threads)