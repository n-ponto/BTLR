"""
Script to load the csv files with the file names of the files in a dataset and 
preprocess them into numpy arrays. The numpy arrays are saved to disk.
"""
import argparse
import os
from parameters import FileParams as fp
from preprocessing.dataloader import DataLoader
from parameters import parameters as params, AudioParams
import numpy as np

DEFAULT_PARAMS = 'mycroft'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Dataset to convert')
parser.add_argument('params', nargs='?', default=DEFAULT_PARAMS,
                    help='Parameters to use for preprocessing')
args = parser.parse_args()


dataset: str = args.dataset
datasets = ['train', 'test', 'val']
assert (
    dataset in datasets), f'Invalid dataset {dataset}. Must be one of {datasets}'

assert (
    args.params in params), f'Invalid parameters {args.params}. Must be one of {params.keys()}'
audioParameters: AudioParams = params[args.params]

# Create the save directory if it does not exist
directory = os.path.join(fp.data_dir, f'np_data_{args.params}_params')
if not os.path.exists(directory):
    print(f'Creating directory {directory}')
    os.makedirs(directory)

# Load the files paths from the csv file
csv_path = os.path.join(fp.data_dir, f'{dataset.lower()}_files.csv')
data_loader = DataLoader(csv_path, ap=audioParameters)
X, y = data_loader.load_data()
print(f'Loaded {len(X)} samples from {csv_path}')


x_path = os.path.join(directory, f'{dataset}_x.npy')
y_path = os.path.join(directory, f'{dataset}_y.npy')


def delete_file_if_exists(path):
    if os.path.exists(path):
        print(f'Deleting {path}')
        os.remove(path)


delete_file_if_exists(x_path)
delete_file_if_exists(y_path)

# Save the data
np.save(x_path, X)
np.save(y_path, y)

print(f'\nSaved {len(X)} samples to {dataset}_x.npy and {dataset}_y.npy')
