from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import sonnet as snt
import preprocess_rgb

import i3d

_IMAGE_SIZE = 224

_BATCH_SIZE = 16
EPOCHS = 100

FRAME_NUM = 125

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map_sr.txt'
IMAGENET_NUM_CLASSES = 400
SR_NUM_CLASSES = 3

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    sr_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, FRAME_NUM, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            IMAGENET_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        m5c, c = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=0.5)

    rgb_variable_map = {}

    for variable in tf.global_variables():
        print(variable)
        if variable.name.split('/')[0] == 'RGB':
            print(variable)
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        tf.logging.info('RGB checkpoint restored')
        print(sess.run('RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0'))
        graph = tf.get_default_graph()
        net = graph.get_tensor_by_name("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0")
        net = tf.stop_gradient(net)
        net_shape = net.get_shape().as_list()
        print(net_shape)

        # add fine-tuning layers
        net = tf.nn.avg_pool3d(net, ksize=[1, 1, 1, 1, 1],
                                         strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        net = tf.nn.dropout(net, 0.5)
        logits = i3d.Unit3D(output_channels=SR_NUM_CLASSES,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=True)
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        predictions = tf.nn.softmax(averaged_logits)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=sr_classes, logits=logits)
        optimizer = tf.train.AdamOptimizer(0.0001)
        train_op = optimizer.minimize(loss)

        dataset, num_batches = preprocess_rgb.train_test_split(_BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()

        for i in range(num_batches):
            images, labels = iterator.get_next()
            init_op = iterator.initializer
            inputs = {'images': images, 'labels': labels, 'iterator_init_op': init_op}
            _, loss_val = sess.run([train_op, loss])


if __name__ == '__main__':
    tf.app.run(main)
