import tensorflow as tf

def map_func(example):

    feature_map = {
        'data': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    data = tf.io.decode_raw(parsed_example["data"], out_type=tf.uint8)
    data = tf.reshape(data, [32, 32, 3])
    label = parsed_example["label"]
    return data, label


model = tf.keras.Sequential([
        tf.keras.applications.resnet_v2.ResNet50V2(weights=None, input_shape=(32, 32, 3), include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])

BATCH_SIZE = 16

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

dataset = tf.data.TFRecordDataset(["./cifar-10-train-data.tfrecords"])
dataset = dataset.map(map_func)
dataset = dataset.batch(32)

val_dataset = tf.data.TFRecordDataset(["./cifar-10-test-data.tfrecords"])
val_dataset = val_dataset.map(map_func)
val_dataset = val_dataset.batch(32)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()


def train_step(data, label):
    print(1)
    with tf.GradientTape() as tape:
        data = tf.cast(data, tf.float32)
        predictions = model(data, training=True)
        loss = loss_obj(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, predictions)


def test_step(data, label):
    predictions = model(data, training=False)
    t_loss = loss_obj(label, predictions)
    test_loss(t_loss)
    test_accuracy(label, predictions)


if __name__ == '__main__':

    for epoch in range(5):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for train_data, train_labels in dataset:
            train_data = train_data.numpy()
            train_data = train_data / 255.0
            train_step(train_data, train_labels)

        for test_data, test_labels in val_dataset:
            test_data = test_data.numpy()
            test_data = test_data / 255.0
            test_step(test_data, test_labels)


        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )



