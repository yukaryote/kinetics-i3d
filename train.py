import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import logging
import sys

sys.path.append("/Users/isabellayu/kinetics-i3d/")

from preprocess.preprocess_rgb import train_test_split
from model.utils import Params
from model.utils import set_logger
from model.train_util import train_and_evaluate
from model.transfer_rgb import model_fn

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/Users/isabellayu/sr-dataset/videos/subclips',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    with tf.device("/GPU:0"):
        # Set the random seed for the whole graph for reproducible experiments
        tf.debugging.set_log_device_placement(True)
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
        logging.info("Creating the datasets...")

        # Create the two iterators over the two datasets
        train_inputs = train_test_split(True, params, args.data_dir)
        val_inputs = train_test_split(False, params, args.data_dir)

        # Define the model
        logging.info("Creating the model...")
        train_model_spec = model_fn('train', train_inputs, params)
        eval_model_spec = model_fn('eval', val_inputs, params)

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(train_model_spec, eval_model_spec,
                           args.model_dir, params, args.restore_from)
