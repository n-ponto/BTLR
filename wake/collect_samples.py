"""
Script to collect samples for training the wake word model.

Three modes are supported:
    - enter: Press enter to save a sample
    - threshold: Automatically save samples above a certain threshold
    - noise: Collect samples of noise to use as negative training samples
"""
import argparse
from audio_collection import SampleCollector, COLLECTION_MODES
from parameters import DEFAULT_AUDIO_PARAMS, FileParams

parser = argparse.ArgumentParser()
parser.add_argument('mode', help=f'Mode to collect samples {COLLECTION_MODES}')
parser.add_argument('-n', '--negative', action='store_true', help='If the samples should go into the negative folder')
args = parser.parse_args()
sample_collector = SampleCollector(DEFAULT_AUDIO_PARAMS, FileParams, args.mode, args.negative)
sample_collector.run()
