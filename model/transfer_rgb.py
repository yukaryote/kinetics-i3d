from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import sys
import time
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append('/Users/isabellayu/kinetics-i3d/')
print(sys.path)
from original_i3d import i3d
from preprocess.rgb_preprocess import train_test_split
from model.utils import Params
from model.utils import set_logger
from model.train_util import train_and_evaluate
from model.transfer_rgb import model_fn

_IMAGE_SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/storage1/sr-kinetics-data/subclips',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
tf.debugging.set_log_device_placement(True)

_CHECKPOINT_PATHS = {
    'rgb': '/home/isabellayu/kinetics-i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '/home/isabellayu/kinetics-i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '/home/isabellayu/kinetics-i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/home/isabellayu/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/home/isabellayu/kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = '../data/label_map_sr.txt'
IMAGENET_NUM_CLASSES = 400
SR_NUM_CLASSES = 3


def make_model(is_training):
    rgb_input = tf.placeholder(tf.float32, shape=(None, 25, 224, 224, 3))
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
        # add fine-tuning layers
    net = tf.nn.avg_pool3d(net, ksize=[1, 1, 1, 1, 1],
                           strides=[1, 1, 1, 1, 1], padding=snt.VALID)

    with tf.variable_scope("RGB"):
        with tf.variable_scope("Dense"):
            net = tf.nn.dropout(net, 0.5, name="dropout")
            logits = i3d.Unit3D(output_channels=3,
                                kernel_shape=[1, 1, 1],
                                activation_fn=None,
                                use_batch_norm=False,
                                use_bias=True,
                                name='Conv3d_0c_1x1')(net, is_training=is_training)
            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
            averaged_logits = tf.reduce_mean(logits, axis=1)
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
    iterator_init_op = inputs['iterator_init_op']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # Compute the output distribution of the model and the predictions
    with tf.variable_scope("RGB", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("Dense", reuse=reuse):
            logits = make_model(is_training)
            predictions = tf.argmax(logits)
            to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="RGB/Dense") 

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    # Only train fine-tuning layers
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, var_list=to_train, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits)),
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
    print(labels.shape, predictions.shape)

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
    model_spec['iterator_init_op'] = iterator_init_op

    if is_training:
        model_spec['train_op'] = train_op
    return model_spec


def train(train_model, val_model, params, train_dataset, val_dataset):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)

        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="RGB/Dense")
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = optimizer.minimize(loss, var_list=to_train, global_step=global_step)
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        for epoch in range(params.num_epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (y_batch_train, x_batch_train) in enumerate(train_dataset):
                logits, loss_value, global_step_val = sess.run([train_op, loss, global_step], feed_dict={x:x_batch_train})
                predictions = tf.argmax(logits)
                loss = tf.losses.sparse_softmax_cross_entropy(labels=y_batch_train, logits=logits)
                loss_value = loss(y_batch_train, logits)

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

                # Log every 200 batches.
                if step % 10 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * params.batch_size))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = val_model(x_batch_val, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))


if __name__ == "__main__":
    tf.set_random_seed(230)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    tf.logging.info("Creating the datasets...")
    train_data = train_test_split(True, params, args.data_dir)
    val_data = train_test_split(False, params, args.data_dir)
    # Define the model
    tf.logging.info("Creating the model...")
    with tf.variable_scope("RGB"):
        train_model = model_fn("train", train_data, params)
    with tf.variable_scope("RGB", reuse=True):
        val_model = model_fn("eval", val_data, params)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    with tf.device("/device:XLA_GPU:0"):
        train(train_model, val_model, params, train_data, val_data)