"""
Script to collect samples for training the wake word model.

Three modes are supported:
    - enter: Press enter to save a sample
    - threshold: Automatically save samples above a certain threshold
    - noise: Collect samples of noise to use as negative training samples
"""

from audio_collection import SampleCollector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'mode', help='Mode to collect samples in [enter, threshold, noise]')
parser.add_argument('-n', '--negative', action='store_true')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
sample_collector = SampleCollector(args.mode, args.negative, args.overwrite)
sample_collector.run()
