"""
Script to load the csv files with the file names of the files in a dataset and 
preprocess them into numpy arrays. The numpy arrays are saved to disk.
"""
import argparse
import os
import numpy as np

from context import parameters, preprocessing

DEFAULT_PARAMS = 'mycroft'
DATASETS = ['train', 'test', 'val']


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Dataset to convert')
parser.add_argument('params', nargs='?', default=DEFAULT_PARAMS,
                    help='Parameters to use for preprocessing')
args = parser.parse_args()

# Check the dataset argument is valid
dataset: str = args.dataset
assert (
    dataset in DATASETS), f'Invalid dataset {dataset}. Must be one of {DATASETS}'

# Check that the parameter argument is valid
assert (
    args.params in parameters.different_parameters), f'Invalid parameters {args.params}. Must be one of {parameters.different_parameters.keys()}'
audioParameters: parameters.AudioParams = parameters.different_parameters[args.params]

# Create the save directory if it does not exist
directory = os.path.join(parameters.FileParams.data_dir,
                         f'np_data_{args.params}_params')
if not os.path.exists(directory):
    print(f'Creating directory {directory}')
    os.makedirs(directory)

# Load the files paths from the csv file
csv_path = os.path.join(parameters.FileParams.data_dir,
                        f'{dataset.lower()}_files.csv')
data_loader = preprocessing.DataLoader(csv_path, ap=audioParameters)
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
