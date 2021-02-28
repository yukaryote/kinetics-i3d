from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    NUM_CLASSES = 3

    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(
            rgb_input, is_training=True, dropout_keep_prob=0.5)

    rgb_variable_map = {}
    model_predictions = tf.nn.softmax(rgb_input)

    for variable in tf.global_variables():
        print(variable)
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        tf.logging.info('RGB checkpoint restored')
        rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
        tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
        feed_dict[rgb_input] = rgb_sample


if __name__ == '__main__':
    tf.app.run(main)
