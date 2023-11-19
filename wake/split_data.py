"""
Searches the data directory for all files and splits them into train, 
validation, and test sets by saving the file names to csv files.
"""

from preprocessing.datasplitter import DataSplitter
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1

CACHE_NAME = './data/.duration_cache.pkl'

# Directories of data created locally
MY_DIRECTORIES = ['noise', 'pos', 'neg']


ds = DataSplitter(MY_DIRECTORIES, (TRAIN_PERCENT,
                  VAL_PERCENT, TEST_PERCENT), CACHE_NAME)
ds.split_data()
