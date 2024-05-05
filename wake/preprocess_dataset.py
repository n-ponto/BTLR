"""
Script to load the csv files with the file names of the files in a dataset and 
preprocess them into numpy arrays. The numpy arrays are saved to disk.
"""
import argparse
import os
import numpy as np
from preprocessing import DATASETS, DataLoader
from parameters import FileParams as FP

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help=f'Dataset to convert, must be one of {DATASETS}')
args = parser.parse_args()
dataset: str = args.dataset
assert (dataset in DATASETS), f'Invalid dataset {dataset}. Must be one of {DATASETS}'

# Load the files paths from the csv file
csv_path = os.path.join(FP.data_dir, f'{dataset.lower()}_files.csv')
data_loader = DataLoader(csv_path)
X, y = data_loader.load_data()
print(f'Loaded {len(X)} samples from {csv_path}')

x_path = os.path.join(FP.data_dir, f'{dataset}_x.npy')
y_path = os.path.join(FP.data_dir, f'{dataset}_y.npy')

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
