from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import sys

sys.path.append('/Users/isabellayu/kinetics-i3d/')
print(sys.path)
from original_i3d import i3d

_IMAGE_SIZE = 224

FRAME_NUM = 125

_CHECKPOINT_PATHS = {
    'rgb': '/Users/isabellayu/kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '/Users/isabellayu/kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '/Users/isabellayu/kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/Users/isabellayu/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/Users/isabellayu/kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = '../data/label_map_sr.txt'
IMAGENET_NUM_CLASSES = 400
SR_NUM_CLASSES = 3


def make_model(is_training, input):
    images = input["images"]
    rgb_input = images
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
            IMAGENET_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
        m5c, c = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=0.5)

    rgb_variable_map = {}

    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable

    for key, value in rgb_variable_map.items():
        print(key, value)
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    with tf.Session() as sess:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        graph = tf.get_default_graph()
        net = graph.get_tensor_by_name("RGB/inception_i3d/Mixed_5c/Branch_3/Conv3d_0b_1x1/batch_norm/beta:0")
    with tf.variable_scope("Dense"):
        # add fine-tuning layers
        net = tf.nn.avg_pool3d(net, ksize=[1, 1, 1, 1, 1],
                               strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        net = tf.nn.dropout(net, 0.5, name="dropout")
        logits = i3d.Unit3D(output_channels=3,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=is_training)
        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        print(averaged_logits)
        for v in tf.trainable_variables():
            print("trainable: ", v)
    return averaged_logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.
        Args:
            mode: (string) can be 'train' or 'eval'
            inputs: (dict) contains the inputs of the graph (features, labels...)
                    this can be `tf.placeholder` or outputs of `tf.data`
            params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
            reuse: (bool) whether to reuse the weights
        Returns:
            model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
        """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # Compute the output distribution of the model and the predictions
    logits = make_model(is_training, inputs)
    predictions = tf.argmax(logits)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    # Only train fine-tuning layers
    to_train = [v for v in tf.trainable_variables() if v.name.startswith("Dense")]
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, var_list=to_train, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    # TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, SR_NUM_CLASSES):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op
    return model_spec
