"""
Script to load the csv files with the file names of the files in a dataset and 
preprocess them into numpy arrays. The numpy arrays are saved to disk.
"""
import argparse
import os
from parameters import FileParams as fp
from preprocessing.dataloader import DataLoader
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Dataset to convert')
args = parser.parse_args()

dataset: str = args.dataset
datasets = ['train', 'test', 'val']
assert(dataset in datasets), f'Invalid dataset {dataset}. Must be one of {datasets}'

# Load the files paths from the csv file
csv_path = os.path.join(fp.data_dir, f'{dataset.lower()}_files.csv')
data_loader = DataLoader(csv_path)
X, y = data_loader.load_data()
print(f'Loaded {len(X)} samples from {csv_path}')

x_path = os.path.join(fp.data_dir, f'{dataset}_x.npy')
y_path = os.path.join(fp.data_dir, f'{dataset}_y.npy')

def delete_file_if_exists(path):
    if os.path.exists(path):
        print(f'Deleting {path}')
        os.remove(path)
    else:
        print(f'{path} does not exist')

delete_file_if_exists(x_path)
delete_file_if_exists(y_path)

# Save the data 
np.save(x_path, X)
np.save(y_path, y)

print(f'\nSaved {len(X)} samples to {dataset}_x.npy and {dataset}_y.npy')