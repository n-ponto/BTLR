import os
import random
import csv
from tqdm import tqdm
from wake.parameters import FileParams as fp
from .durationcache import DurationCache

VALID_FILE_TYPES = ['.wav', '.mp3']

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

RANDOM_SEED = 0


class DataSplitter:
    """
    Searches the data directory for all files and splits them into train,
    validation, and test sets by saving the file names to csv files.
    """

    def __init__(self, my_directories: list[str], percentages: (float, float, float), cache_location: str):
        """
        Initializes the DataSplitter
        Args:
            my_directories: list of directories created locally
            percentages: tuple of floats (train, val, test)
            cache_location: path to the duration cache
        """
        assert (sum(percentages) == 1)
        self.my_directories = my_directories
        self.train_percent, self.val_percent, self.test_percent = percentages
        self.durationCache = DurationCache(cache_location)

    def split_data(self):
        random.seed(RANDOM_SEED)

        # Split each of the subdirectories into train, val, and test datasets
        subdir_datasets = self._get_subdir_datasets()

        # Figure out the duration of each of the train sets
        train_durations = self._get_train_durations(subdir_datasets)
        print()

        self._multiply_train_sets(subdir_datasets, train_durations)

        # Get the positive and negative datasets
        # NOTE: only augment data
        positive_dataset = subdir_datasets[fp.pos_sample_dir]
        negative_datasets = [
            (x, k in self.my_directories) for k, x in subdir_datasets.items() if k != fp.pos_sample_dir]

        # Save the datasets
        self._save_dataset_splits(positive_dataset, negative_datasets)

    def _get_subdir_datasets(self):
        """
        Creates a mapping of each subdirectory within the data directory to a
        tuple of lists of file paths (train, val, test)
        Returns:
            dict of {subdirectory -> (train, val, test)}
        """
        # Switch to the data directory
        os.chdir(fp.data_dir)
        subdirs = os.listdir(fp.data_dir)
        subdir_datasets = {}

        # Split the files into train, val, and test datasets
        for subdir in subdirs:
            if not os.path.isdir(subdir):
                print(f'ignoring: {subdir} is not a directory')
                continue

            files = os.listdir(subdir)
            files = [os.path.join(subdir, x) for x in files]

            # Only include files with valid extensions
            files = [x for x in files if os.path.isfile(x) and
                     any([x.lower().endswith(y) for y in VALID_FILE_TYPES])]

            datasets = self._split_files(files)
            if datasets is None:
                print(f'ignoring: {subdir} has no valid files')
                continue
            subdir_datasets[subdir] = datasets
        return subdir_datasets

    def _get_train_durations(self, subdir_datasets: dict[str, (list, list, list)]) -> dict[str, float]:
        """
        Get the total duration of audio files within each subdirectory.
        Args:
            subdir_datasets: dict of subdirectory -> (train, val, test)
        Returns:
            dict of subdirectory -> total duration of train set
        """
        train_durations = {}
        for subdir, datasets in subdir_datasets.items():
            train_dataset = datasets[0]
            print(f'\nGetting duration of {subdir} train set...')
            total_duration = self._get_total_duration(
                train_dataset, self.durationCache)
            print(f'{subdir} total duration {round(total_duration)} seconds')
            train_durations[subdir] = total_duration
        return train_durations

    def _multiply_train_sets(self, subdir_datasets: dict[str, (list, list, list)], train_durations: dict[str, float]) -> None:
        """
        Multiply the training sets created locally to match the other datasets.
        NOTE: assuming only pos, neg, and noise local sets, then each subdir
              will end up being 25% of the total training data
        Args:
            subdir_datasets: dict of subdirectory -> (train, val, test)
            train_durations: dict of subdirectory -> total duration of train set
        """
        # Get total duration of all train
        total_train_duration = sum(train_durations.values())
        print(f'Total train duration {round(total_train_duration)} seconds')

        # Get total duration of train sets not created locally
        total_other_duration = sum(
            [x for k, x in train_durations.items() if k not in self.my_directories])
        for subdir, (train, val, test) in subdir_datasets.items():
            if subdir not in self.my_directories:
                continue
            duration = train_durations[subdir]
            percent_of_total = duration / total_train_duration if total_train_duration > 0 else 0
            print(
                f'\n{subdir} makes up {round(percent_of_total * 100, 2)}% of total')
            multiplier = round(total_other_duration / duration)
            print(f'{subdir} multiplier: {multiplier}')
            subdir_datasets[subdir] = (train * multiplier, val, test)
        print(f'Total other duration {round(total_other_duration)} seconds')

    def _get_total_duration(self, files: list[str], dc: DurationCache):
        """
        Gets the total duration of the files in seconds.
        Args:
            files: list of file paths
            dc: duration cache
        Returns:
            The total duration of the files in seconds
        """
        total_duration = 0
        for file in tqdm(files):
            total_duration += dc.get_duration(file)
        return total_duration

    def _split_files(self, files: list[str]):
        """
        Takes a list of files, randomizes, and splits into datasets
        Args:
            files: list of file paths
        Returns:
            A tuple of lists of file paths (train, val, test)
        """
        if len(files) == 0:
            return None
        random.shuffle(files)
        # Split into datasets
        count_train = int(len(files) * self.train_percent)
        count_val = int(len(files) * self.val_percent)
        train_files = files[:count_train]
        val_files = files[count_train:count_train+count_val]
        test_files = files[count_train+count_val:]
        return train_files, val_files, test_files

    def _save_dataset_splits(self, pos_dataset: (list, list, list), neg_datasets: list[((list, list, list), bool)]):
        """
        Saves the datasets to csv files.
        Args:
            pos_dataset: tuple of lists of file paths (train, val, test)
            neg_dataset: tuple of tuples (train, val, test), bool
                The bool indicates whether the dataset should be augmented
        """
        directories = [TRAIN_DIR, VAL_DIR, TEST_DIR]

        assert (len(pos_dataset) == len(directories))
        assert (len(neg_datasets[0][0]) == len(directories))
        assert (type(neg_datasets[0][0]) ==
                tuple), f'{type(neg_datasets[0][0])}'
        assert (type(neg_datasets[0][1]) == bool)
        assert (type(neg_datasets[0][0][0]) == list)

        for i, dir in enumerate(directories):
            # Get the positive and negative files corresponding to the dataset type (either train, val, or test)
            pos_files = pos_dataset[i]
            neg_files = []
            for datasets, augment in neg_datasets:
                data = datasets[i]
                assert (type(data) == list)
                neg_files.extend([(f, augment) for f in data])

            assert (type(neg_files) == list)

            # Columns are: file_path, label, augment
            allow_augment = dir == TRAIN_DIR  # Only augment the training set
            pos_rows = [[x, 1, int(True and allow_augment)] for x in pos_files]
            neg_rows = [[x, 0, int(a and allow_augment)] for x, a in neg_files]
            all_rows = pos_rows + neg_rows
            random.shuffle(all_rows)

            filename = os.path.join(fp.data_dir, f'{dir}_files.csv')

            # Delete the file if it already exists
            if os.path.isfile(filename):
                print(f'Deleting {filename}')
                os.remove(filename)

            print(f'Saving {len(all_rows)} file names to {filename}')
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_rows)
