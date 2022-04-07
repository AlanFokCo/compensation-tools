import numpy as np

TRAIN_DATA_PATHS = "../imagenet_train_data/train_data_batch_%d.npz"
VAL_DATA_PATH = "../val_data.npz"
DATASET_SUM = 1281160
DATASET_PER = 128116


def normalization(data):
    for index in range(len(data)):
        data[index].resize(64, 64, 3)
    return data


def load_data(path):
    data = np.load(path)
    result = []

    labels = data.f.labels
    data = data.f.data
    for index in range(len(data)):
        temp = data[index]
        temp.resize(64, 64, 3)
        result.append(temp)
    return np.array(result), labels


def load_val_data(partition, index):
    # partition should >= 10
    if partition < 10:
        raise Exception("partition < 10, should >= 10.")
    data, labels = load_data(VAL_DATA_PATH)
    length = len(data)
    per_num = length // partition
    start = index * per_num
    end = (index + 1) * per_num
    data = data[start:end]
    labels = labels[start:end]
    return data, labels


def load_train_data(partition, index):

    # partition should >= 10
    if partition < 10:
        raise Exception("partition < 10, should >= 10.")
    per_num = DATASET_SUM // partition
    start = index * per_num
    end = (index + 1) * per_num
    pre = start // DATASET_PER
    after = end // DATASET_PER
    if pre == after:
        data, labels = load_data(TRAIN_DATA_PATHS % pre)
        i = start % DATASET_PER
        data = data[i:i + per_num]
        labels = labels[i:i + per_num]
        return data, labels
    else:
        data_pre, labels_pre = load_data(TRAIN_DATA_PATHS % pre)
        i = start % DATASET_PER
        data_pre = data_pre[i:]
        labels_pre = labels_pre[i:]
        data_after, labels_after = load_data(TRAIN_DATA_PATHS % after)
        i = end % DATASET_PER
        data_after = data_after[:i]
        labels_after = labels_after[:i]
        return np.append(data_pre, data_after), np.append(labels_pre, labels_after)
