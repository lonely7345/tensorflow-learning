# -*- coding: utf-8 -*-

import tensorflow as tf

sess = tf.Session()


#加载TFRecord文件
# Load TFRecord
tf_record_filename_queue = tf.train.string_input_producer(
    ["./output/training-image.tfrecord"])

# Notice the different record reader, this one is designed to work with TFRecord files which may
# have more than one example in them.
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

# The label and image are stored as bytes but could be stored as int64 or float64 values in a
# serialized tf.Example protobuf.
tf_record_features = tf.parse_single_example(
    tf_record_serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })

# Using tf.uint8 because all of the channel information is between 0-255
tf_record_image = tf.decode_raw(
    tf_record_features['image'], tf.uint8)

 
# Use real values for the height, width and channels of the image because it's required
# to reshape the input.

tf_record_label = tf.cast(tf_record_features['label'], tf.string)

# setup-only-ignore
sess.close()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

 
# Check that the label is still 0b00000001.
print sess.run(tf_record_label)
print sess.run(tf_record_image)

# setup-only-ignore
tf_record_filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)
