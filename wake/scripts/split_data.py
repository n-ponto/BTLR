"""
Searches the data directory for all files and splits them into train, 
validation, and test sets by saving the file names to csv files.
"""

from context import preprocessing
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1

PERCENTAGES = (TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

CACHE_NAME = '../data/.duration_cache.pkl'

# Directories of data created locally
MY_DIRECTORIES = ['noise', 'pos', 'neg']


ds = preprocessing.DataSplitter(MY_DIRECTORIES, PERCENTAGES, CACHE_NAME)
ds.split_data()
