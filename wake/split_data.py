"""
Searches the data directory for all files and splits them into train, 
validation, and test sets by saving the file names to csv files.
"""
from preprocessing import DataSplitter
from parameters import FileParams as FP

TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1
PERCENTAGES = (TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

# Directories of data created locally (as opposed to downloaded)
MY_DIRECTORIES = [FP.pos_sample_name, FP.neg_sample_name, FP.noise_sample_name]

data_splitter = DataSplitter(MY_DIRECTORIES, PERCENTAGES)
data_splitter.split_data()
