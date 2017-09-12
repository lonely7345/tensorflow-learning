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

# Reuse the image from earlier and give it a fake label
image_label = b'\x01'  # Assume the label data is in a one-hot representation (00000001)

# Convert the tensor into bytes, notice that this will load the entire image file
image_loaded = sess.run(image)
image_bytes = image_loaded.tobytes()
image_height, image_width, image_channels = image_loaded.shape

# Export TFRecord
writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")

# Don't store the width, height or image channels in this Example file to save space but not required.
example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))

# This will save the example to a text file tfrecord
writer.write(example.SerializeToString())
writer.close()

 