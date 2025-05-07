#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.


import sys
import os

# Get the absolute path to the parent directory of 'eztorch'
eztorch_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to the Python path if it's not already there
if eztorch_parent_dir not in sys.path:
    print('added to path')
    sys.path.insert(0, eztorch_parent_dir)

    

import argparse
import logging
from pathlib import Path
from typing import Dict

from spivak.application.test_utils import test, SharedArgs, \
    translate_dataset_type_to_custom
from spivak.data.dataset_splits import SPLIT_KEY_UNLABELED
from spivak.data.video_io import list_video_paths
from spivak.feature_extraction.extraction import extract_features_from_videos, \
    EXTRACTOR_TYPE_RESNET_TF2, create_feature_extractor


class Args:
    INPUT_VIDEOS_DIR = "input_dir"
    FEATURES_DIR = "features_dir"
    RESULTS_DIR = "results_dir"
    LABELS_DIR = "labels_dir"
    MODEL_PATH = "model"
    FEATURES_MODELS_DIR = "features_models_dir"
    FEATURES = "features"
    CONFIG_DIR = "config_dir"
    SPLITS_DIR = "splits_dir"


def main() -> None:
    args = _get_command_line_arguments()

    try:
        print("0 args.feature_name", args.feature_name)
    except:
        print('0 args.feature_name', 'None')


    logging.getLogger().setLevel(logging.DEBUG)
    input_dir = Path(args[Args.INPUT_VIDEOS_DIR])
    if input_dir.is_dir():
        raise ValueError(f"Input directory failed is_dir(): {input_dir}")
    
    features_dir = Path(args[Args.FEATURES_DIR])
    results_dir = Path(args[Args.RESULTS_DIR])
    features_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor = create_feature_extractor(
        args[Args.FEATURES], Path(args[Args.FEATURES_MODELS_DIR]))
    video_paths = [input_dir]
    extract_features_from_videos(video_paths, features_dir, feature_extractor)
    # Set up prediction run by loading existing model arguments and
    # overwriting them for the current run.
    # TODO: allow this to work for "dense_delta" model by taking
    #  two input models in MODEL_PATH and running inference twice in a row,
    #  once for each model. For now, can just run it twice with the same
    #  results folder (first with just the confidence model, then with the
    #  dense_delta one).
    shared_args = SharedArgs.load(args[Args.MODEL_PATH])
    shared_args.model = args[Args.MODEL_PATH]
    shared_args.results_dir = results_dir
    shared_args.features_dir = str(features_dir)
    shared_args.config_dir = args[Args.CONFIG_DIR]
    shared_args.dataset_type = translate_dataset_type_to_custom(
        shared_args.dataset_type)
    shared_args.labels_dir = args[Args.LABELS_DIR]
    shared_args.splits_dir = args[Args.SPLITS_DIR]
    shared_args.test_split = SPLIT_KEY_UNLABELED
    # Don't run evaluation, just prediction.
    shared_args.evaluate = 0
    # Run prediction code on the generated features.
    
    test(shared_args)


def _get_command_line_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--" + Args.INPUT_VIDEOS_DIR, help='Input directory containing videos',
        required=True)
    parser.add_argument(
        "--" + Args.RESULTS_DIR, help="Output directory", required=True)
    parser.add_argument(
        "--" + Args.MODEL_PATH, help="Model directory", required=True)
    parser.add_argument(
        "--" + Args.FEATURES_MODELS_DIR, required=True,
        help="Directory containing models used for extracting video features")
    parser.add_argument(
        "--" + Args.FEATURES_DIR, required=True,
        help="Directory in which to store intermediate video features")
    parser.add_argument(
        "--" + Args.LABELS_DIR, required=False,
        help="Directory containing label files, if available")
    parser.add_argument(
        "--" + Args.CONFIG_DIR, type=str, required=True,
        help="Directory containing a set of config files",)
    parser.add_argument(
        "--" + Args.SPLITS_DIR, type=str, required=True,
        help="Directory for storing the generated split files")
    parser.add_argument(
        "--" + Args.FEATURES, required=False,
        help="What type of features to use", default=EXTRACTOR_TYPE_RESNET_TF2,
        choices=[EXTRACTOR_TYPE_RESNET_TF2])
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    main()
