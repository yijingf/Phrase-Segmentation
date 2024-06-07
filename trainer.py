# -*- coding: utf-8 -*-
import os
import argparse
import warnings
from glob import glob

from train import train_model
from configuration import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='exp_unsupervised')
    parser.add_argument('--ds_path', type=str, default="./data",
                        help='Path to training dataset')
    parser.add_argument('--model_name', type=str,
                        default='test', help="Model name, defaults to 'test'")
    parser.add_argument('--feat_id', type=str, default="mel",
                        help="Feature used, defaults to 'mel'")
    parser.add_argument('--n_epoch', type=int, default=100,
                        help="Number of training epochs, defaults to 100")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size, defaults to 32")
    parser.add_argument('--nb_workers', type=int, default=8,
                        help="Number of workers, defaults to 8")
    args, _ = parser.parse_known_args()

    # Input and Output
    config.ds_path = args.ds_path
    config.feature_types = ['mel', 'chroma']

    feat_list = sorted(glob(os.path.join(args.ds_path, "features/mel/*.npy")))
    tracklist = []
    for feat_file in feat_list:
        audio_file = feat_file.replace("features/mel", "audio")
        audio_file = audio_file.replace(".npy", ".wav")
        tracklist.append(audio_file)
    config.tracklist = tracklist

    # Model
    config.model_name = args.model_name
    config.architecture = 'EmbedNet'
    config.embedding.n_embedding = 512

    # Training Setup
    config.training_strategy = 'triplet_features'
    config.seg_algo = 'scluster'

    # Training Param
    config.epochs = args.n_epoch
    config.batch_size = args.batch_size
    config.nb_workers = args.nb_workers
    config.no_cuda = False
    config.resume = False
    config.learning_rate = 1e-3
    config.margin = 0.1

    warnings.simplefilter('ignore')
    train_model(config)
