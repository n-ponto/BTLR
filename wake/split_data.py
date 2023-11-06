"""
Searches the data directory for all files and splits them into train, 
validation, and test sets by saving the file names to csv files.
"""

from parameters import FileParams as fp
import os
import random
import csv
import wave
import librosa
from parameters import MycroftParams as ap
from tqdm import tqdm
import pickle

TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1

# Will duplicate positive files to reach this ratio of positive to negative samples
TARGET_POS_RATIO = 0.35

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

SHUFFLE = True

VALID_FILE_TYPES = ['.wav', '.mp3']

MIN_FILE_DURATION = ((ap.n_features - 1) * ap.hop_samples + ap.window_samples) / ap.sample_rate
assert(MIN_FILE_DURATION == 1.5)

CACHE_NAME ='.duration_cache.pkl'

class DurationCache:
    """
    Caches the duration of files so we don't have to keep loading them
    """
    duration_cache: dict  # file_path -> duration

    def __init__(self):
        self.duration_cache = {}
        self.load_cache()
       
    def load_cache(self):
        print('Loading cache')
        if os.path.isfile(CACHE_NAME):
            print('Is file')
            with open(CACHE_NAME, 'rb') as f:
                self.duration_cache = pickle.load(f)
                print('Loaded cache')
        else:
            print('Creating new duration cache')
            self.duration_cache = {}

    def __del__(self):
        self.save()

    def save(self):
        with open(CACHE_NAME, 'wb') as f:
            pickle.dump(self.duration_cache, f)

    def get_duration(self, file_path: str):
        """
        Gets the duration of the file in seconds
        Args:
            file_path: The path to the file
        Returns:
            The duration of the file in seconds
        """
        if file_path in self.duration_cache:
            return self.duration_cache[file_path]

        if file_path.endswith('.wav'):
            duration = get_duration_wav(file_path)
        elif file_path.endswith('.mp3'):
            duration = get_duration_mp3(file_path)
        else:
            raise ValueError(f'Cannot get duration of {file_path}')
        duration = max(MIN_FILE_DURATION, duration)
        self.duration_cache[file_path] = duration

def get_duration_wav(file_path: str):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of frames and the frame rate (samples per second)
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        
        # Calculate the duration in seconds
        duration = num_frames / frame_rate
        return duration

def get_duration_mp3(file_path):
    duration_in_seconds = librosa.get_duration(path=file_path)
    return duration_in_seconds

def get_pos_files():
    """
    Loads the list of file names from the directory with positive samples
    """
    pos_dir = os.path.join(fp.data_dir, fp.pos_sample_dir)
    pos_files = os.listdir(pos_dir)  # Everything in directory
    pos_files = [os.path.join(pos_dir, x)
                 for x in pos_files]  # Add full file path
    pos_files = [x for x in pos_files if os.path.isfile(x)]  # Only keep files
    print(f'Found {len(pos_files)} positive files.')
    return pos_files


def get_other_files():
    """
    Gets the list of file names from all directories that aren't positive samples
    """
    other_files = []
    for sub_dir in os.listdir(fp.data_dir):
        if sub_dir == fp.pos_sample_dir:
            continue
        dir_path = os.path.join(fp.data_dir, sub_dir)
        if not os.path.isdir(dir_path):
            continue
        new_files = os.listdir(dir_path)  # Everything in directory
        new_files = [os.path.join(dir_path, x)
                     for x in new_files]  # Add full file path
        # Only keep files
        new_files = [x for x in new_files if os.path.isfile(x) and any([x.endswith(y) for y in VALID_FILE_TYPES])]
        print(f'\tFound {len(new_files)} files in {sub_dir}')
        other_files.append((sub_dir, new_files))
    print(f'Found {len(other_files)} subdirectories.')
    return other_files


def create_split(files: list[str]):
    """
    Takes a list of files, randomizes, and splits into datasets
    """
    random.shuffle(files)
    # Split into datasets
    count_train = int(len(files) * TRAIN_PERCENT)
    count_val = int(len(files) * VAL_PERCENT)
    train_files = files[:count_train]
    val_files = files[count_train:count_train+count_val]
    test_files = files[count_train+count_val:]
    return train_files, val_files, test_files

def get_pos_multiplier(pos_files, other_files) -> int:
    """
    Finds the ratio of positive to negative samples and returns the multiplier
    so that the positive samples will be duplicated to reach the target ratio
    Args:
        pos_files: list of positive sample file paths
        other_files: list of tuples (subdirectory, list_file_paths)
    Returns:
        The multiplier to reach the target ratio
    """

    dc = DurationCache()

    def get_durations(files):
        with_durations = []
        for file in tqdm(files):
            with_durations.append((file, dc.get_duration(file)))
        return with_durations

    # Get durations of the files
    print('Getting pos sample durations')
    pos_files_with_duration = get_durations(pos_files)

    neg_files_with_duration = []
    for subdir, fs in other_files:
        print(f'Getting durations for {subdir}')
        neg_files_with_duration.extend(get_durations(fs))
    
    del dc
    pos_total_duration = sum(d for f, d in pos_files_with_duration)
    print(f'\tTotal duration of positive samples {pos_total_duration}')
    neg_total_duration = sum(d for f, d in neg_files_with_duration)
    print(f'\tTotal duration of negative samples {neg_total_duration}')

    pos_to_neg_ratio = pos_total_duration / neg_total_duration
    print(f'\tRatio of positive to negative duration {pos_to_neg_ratio}')
    multiplier = round(TARGET_POS_RATIO / pos_to_neg_ratio)
    print(f'\tMultiplying positive set by {multiplier} to reach {TARGET_POS_RATIO}')
    return multiplier

def create_datasets():
    """
    Finds all the files (pos and neg) in the data directory and splits into datasets
    Returns: 
        positive and negative datasets, each with lists of train, validation, 
        and test files
    """
    pos_files = get_pos_files()  # list of file paths
    other_files = get_other_files()  # list of tuples (subdirectory, list_file_paths)
    multiplier = get_pos_multiplier(pos_files, other_files)
    
    # Split samples
    pos_dataset = create_split(pos_files)

    # Multiply the training positive samples
    pos_train, pos_val, pos_test = pos_dataset
    pos_dataset = (pos_train * multiplier, pos_val, pos_test)

    # Split each of the subdirectories separately
    neg_dataset = ([], [], [])
    for _, files in other_files:
        train, val, test = create_split(files)
        neg_dataset[0].extend(train)
        neg_dataset[1].extend(val)
        neg_dataset[2].extend(test)
    return pos_dataset, neg_dataset


def main():
    pos_dataset, neg_dataset = create_datasets()
    directories = [TRAIN_DIR, VAL_DIR, TEST_DIR]

    for i, dir in enumerate(directories):
        pos_files = pos_dataset[i]
        neg_files = neg_dataset[i]

        pos_rows = [[x, 1] for x in pos_files]
        neg_rows = [[x, 0] for x in neg_files]
        rows = pos_rows + neg_rows
        random.shuffle(rows)

        filename = os.path.join(fp.data_dir, f'{dir}_files.csv')
        print(f'Saving {len(rows)} file names to {filename}')

        # Delete the file if it already exists
        if os.path.isfile(filename):
            os.remove(filename)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
