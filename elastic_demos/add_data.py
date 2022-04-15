import pickle
import numpy as np
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_single(file):
    data = unpickle(file)

    result = []
    labels = data[b'labels']
    data = data[b'data']
    for index in range(len(data)):
        temp = data[index]
        temp.resize(32, 32, 3)
        result.append(temp)
    return np.array(result), labels


TRAIN_DATA_PATHS = "./cifar_data/data_batch_%d"
TEST_DATA_PATH = "./cifar_data/test_batch"
# train_data = []
# train_labels = []
# #
# # for i in range(1, 6):
# #     temp_data, temp_labels = load_data_single(TRAIN_DATA_PATHS % i)
# #     train_labels.extend(temp_labels)
# #     train_data.extend(temp_data)
# #
# train_data = np.array(train_data)
# train_labels = np.array(train_labels)

data, labels = load_data_single(TEST_DATA_PATH)
print(len(labels))
np.savez("cifar-10-test-data.npz", data=data, labels=labels)
# np.savez("cifar-10-train-data.npz", data=train_data, labels=train_labels)

# w = tf.io.TFRecordWriter("./cifar-10-val-data.tfrecords")
# for i in range(10000):
#     example = tf.train.Example(
#         features=tf.train.Features(
#             feature={
#                 "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].tobytes()])),
#                 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
#             }
#         )
#     )
#     w.write(example.SerializeToString())
#
# w.close()

