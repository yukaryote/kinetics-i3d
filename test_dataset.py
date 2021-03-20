import tensorflow as tf
import numpy as np

test_arr = [[np.zeros(shape=(2, 2), dtype="float32"), 0], [np.ones(shape=(2, 2), dtype="float32"), 1]]


def input_gen():
    for i in range(len(test_arr)):
        label = test_arr[i][1]
        features = test_arr[i][0]
        yield label, features


dataset = tf.data.Dataset.from_generator(input_gen, output_types=(tf.int64, tf.float32)).batch(1)
iterator = dataset.make_initializable_iterator()
x, y = iterator.get_next()
print(x, y)
