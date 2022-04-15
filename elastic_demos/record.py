import numpy as np
import tensorflow as tf


data = np.load("./cifar-10-test-data.npz")

labels = data.f.labels
data = data.f.data

w = tf.io.TFRecordWriter("./cifar-10-test-data.tfrecords")
for i in range(10000):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].tobytes()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
            }
        )
    )
    w.write(example.SerializeToString())

w.close()


def map_func(example):
    feature_map = {
        'data': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed_example = tf.parse_single_example(example, features=feature_map)
    data = tf.decode_raw(parsed_example["data"], out_type=tf.uint8)
    data = tf.reshape(data, [32, 32, 3])
    label = parsed_example["label"]
    return data, label

