"""
Script to collect samples for training the wake word model.

Three modes are supported:
    - enter: Press enter to save a sample
    - threshold: Automatically save samples above a certain threshold
    - noise: Collect samples of noise to use as negative training samples
"""

from context import audio_collection, parameters

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'mode', help='Mode to collect samples in [enter, threshold, noise]')
parser.add_argument('-n', '--negative', action='store_true', help='If the samples should go into the negative folder')
parser.add_argument('-o', '--overwrite', action='store_true', help='If the samples should overwrite existing samples')
args = parser.parse_args()
sample_collector = audio_collection.SampleCollector(parameters.DEFAULT_AUDIO_PARAMS, parameters.FileParams, args.mode, args.negative, args.overwrite)
sample_collector.run()
