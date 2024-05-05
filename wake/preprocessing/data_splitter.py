import os
import random
import csv
from tqdm import tqdm
import librosa
from parameters import FileParams as FP, DEFAULT_AUDIO_PARAMS as AP

VALID_FILE_TYPES = ['.wav', '.mp3']

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"
DATASETS = [TRAIN_DIR, VAL_DIR, TEST_DIR]

RANDOM_SEED = 0  # So the outcome of splitting data is deterministic


class DataSplitter:
    """Searches the data directory for all files and splits them into train,
    validation, and test sets by saving the file names to csv files.

    Expects the data directory to be structured as follows:

    data_dir
    ├── postive_samples (FileParams.pos_sample_name)
    │   ├── postive_sample1.wav
    │   ├── postive_sample2.wav
    │   └── ...
    ├── negative_samples (FileParams.neg_sample_name)
    │   ├── negative_sample1.wav
    │   ├── negative_sample2.wav
    │   └── ...
    ├── noise_samples (FileParams.noise_sample_name)
    │   ├── noise_sample1.wav
    │   ├── noise_sample2.wav
    │   └── ...
    ├── downloaded_dir1 (arbitrary name)
    │   ├── downloaded_file1.wav
    │   └── ...
    └── ... (arbitrary name)
    """

    def __init__(self, my_directories, percentages):
        """Initializes the DataSplitter
        Args:
            my_directories: `list[str]` list of directories created locally
            percentages: tuple of floats (train, val, test)
        """
        assert (sum(percentages) == 1)
        self.my_directories = my_directories
        self.train_percent, self.val_percent, self.test_percent = percentages

    def split_data(self):
        """Splits the data into train, val, and test datasets and saves the
        file names to csv files.
        """
        random.seed(RANDOM_SEED)
        os.chdir(FP.data_dir)  # Switch to the data directory

        # Split each of the subdirectories into train, val, and test datasets
        subdir_datasets = self._create_subdir_datasets()

        # Figure out the number of samples of each of the train sets
        train_samples = self._get_train_samples(subdir_datasets)

        # Use the number of training samples in each subdirectory to multiply
        # the training sets for better data distribution
        self._multiply_train_sets(subdir_datasets, train_samples)

        # Save the datasets
        self._save_dataset_splits(subdir_datasets)

    def _create_subdir_datasets(self) -> dict[str, (list[str], list[str], list[str])]:
        """Creates a mapping of each subdirectory within the data directory to a
        tuple of lists of file paths (train, val, test)
        Returns:
            dict of {subdirectory -> (train, val, test)}
        """
        subdirs = os.listdir()
        subdir_datasets = {}

        def has_valid_extension(x: str):
            """Returns True if the file has a valid audio file extension (like WAV or MP3)."""
            return any([x.lower().endswith(y) for y in VALID_FILE_TYPES])

        # Split the files into train, val, and test datasets
        for subdir in subdirs:
            if not os.path.isdir(subdir):
                print(f'ignoring: {subdir} is not a directory')
                continue

            # Get all the valid files within the subdirectory
            files = os.listdir(subdir)
            # Add the subdirectory to the file path
            files = [os.path.join(subdir, x) for x in files]

            # Only include files with valid extensions
            files = [x for x in files if os.path.isfile(
                x) and has_valid_extension(x)]
            if len(files) == 0:
                print(f'ignoring: {subdir} has no valid files')
                continue
            datasets = self._split_files(files)
            subdir_datasets[subdir] = datasets
        print()
        return subdir_datasets

    def _split_files(self, files: list[str]):
        """Takes a list of files, randomizes, and splits into datasets
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

    def _get_train_samples(self, subdir_datasets: dict[str, (list[str], list[str], list[str])]) -> dict[str, int]:
        """Get the number of training samples that can be created from audio files 
        within each subdirectory. 

        Each positive or negative file corresponds to a single training sample, 
        but a long noise or downloaded file may correspond to multiple training 
        samples based on its length.
        Args:
            subdir_datasets: dict of subdirectory -> (train, val, test)
        Returns:
            dict of subdirectory -> number of training samples in the subdirectory.
        """
        train_samples = {}  # dict mapping subdir -> number of training samples
        for subdir, datasets in subdir_datasets.items():
            train_dataset = datasets[0]

            # Each pos or neg sample corresponds to one training sample
            if subdir in [FP.pos_sample_name, FP.neg_sample_name]:
                # num samples = num files
                train_samples[subdir] = len(train_dataset)
                continue

            # Otherwise, get number samples based on length of audio files
            print(f'\nGetting samples in {subdir} train set...')
            samples = self._get_total_samples(train_dataset)
            print(f'{subdir} total samples {round(samples)}')
            train_samples[subdir] = samples
        return train_samples

    def _get_total_samples(self, files: list[str]) -> int:
        """Gets the total number of training samples in the files. Checks the
        length of each audio file, figures out how many training samples can be
        created from it, and sums them all up for the directory.
        Args:
            files: list of file paths
        Returns:
            The total samples in the files.
        """
        PERCENT_DIR_TO_SAMPLE = 0.25  # Percentage of the files to check for duration
        total_samples = 0
        # Assuming the files are on average the same length, sample a subset
        sample_subset = random.choices(files, k=int(len(files) * PERCENT_DIR_TO_SAMPLE))
        for file_path in tqdm(sample_subset):
            # Get file duration (in seconds)
            duration = librosa.get_duration(path=file_path)
            # Calculate number of training samples based on duration
            count_windows = (duration - AP.window_t) // AP.hop_t + 1
            samples = count_windows // AP.n_features
            if count_windows % AP.n_features != 0:
                samples += 1
            total_samples += samples
        return int(total_samples / PERCENT_DIR_TO_SAMPLE)

    def _multiply_train_sets(self, subdir_datasets: dict[str, (list, list, list)], train_samples: dict[str, int]) -> None:
        """Multiply the training sets created locally to match the other datasets.
        Modifies the training sets in place of `subdir_datasets`.

        For example, if there this count of training samples:
        - positive: 10
        - negative: 20
        - noise: 25
        - downloaded_dir1: 100
        Then the training sets will be multiplied by the following factors:
        - positive: 10
        - negative: 5
        - noise: 4
        So that each subdir contributes 100 samples to training
        (downloaded_dir1 is not multiplied because it is not created locally)

        NOTE: assuming only pos, neg, and noise local sets, then each subdir
              will end up being 25% of the total training data
        Args:
            subdir_datasets: dict of subdirectory -> (train, val, test)
            train_samples: dict of subdirectory -> total samples of train dataset
        """
        # Get total training samples of all subdirectories
        tot_cnt_trn_smpls = sum(train_samples.values())  # just for printing
        print(f'Total training samples: {round(tot_cnt_trn_smpls)}')

        # Get total samples of train sets not created locally
        count_other_samples = sum([x for k, x in train_samples.items() if k not in self.my_directories])
        for subdir, (train, val, test) in subdir_datasets.items():
            if subdir not in self.my_directories:
                continue  # Skip multiplying non-local datasets
            cnt_smpls = train_samples[subdir]
            prcnt_total = cnt_smpls / tot_cnt_trn_smpls if tot_cnt_trn_smpls > 0 else 0
            print(f'\nInitially {subdir} makes up {round(prcnt_total * 100, 2)}% of total')
            
            multiplier = round(count_other_samples / cnt_smpls)
            print(f'{subdir} multiplier: {multiplier}')

            # Multiply training data to match the other non-local datasets
            subdir_datasets[subdir] = (train * multiplier, val, test)
        print(f'Total other duration {round(count_other_samples)} seconds')

    def _save_dataset_splits(self, subdir_datasets: dict[str, (list, list, list)]):
        """Saves the datasets to csv files.
        Args:
            subdir_datasets: dict of subdirectory -> (train, val, test)
        """
        assert (all([type(x) == tuple for x in subdir_datasets.values()]))
        assert (all([len(x) == len(DATASETS) for x in subdir_datasets.values()]))
        assert (all([type(x[0]) == list for x in subdir_datasets.values()]))

        # Get the positive and negative datasets
        positive_dataset = subdir_datasets[FP.pos_sample_name]
        # NOTE: only set augment bool to True for directories created locally
        negative_datasets = [(x, k in self.my_directories)
                             for k, x in subdir_datasets.items() if k != FP.pos_sample_name]

        for i, dataset in enumerate(DATASETS):
            # Get the positive and negative files corresponding to the dataset type (either train, val, or test)
            pos_files = positive_dataset[i]
            neg_files = []
            for datasets, augment in negative_datasets:
                data = datasets[i]
                assert (type(data) == list)
                neg_files.extend([(f, augment) for f in data])

            assert (type(neg_files) == list)

            # Columns are: file_path, label, augment
            allow_augment = dataset == TRAIN_DIR  # Only augment the training set
            pos_rows = [[x, 1, int(True and allow_augment)] for x in pos_files]
            neg_rows = [[x, 0, int(a and allow_augment)] for x, a in neg_files]
            all_rows = pos_rows + neg_rows
            random.shuffle(all_rows)

            filename = f'{dataset}_files.csv'

            # Delete the file if it already exists
            if os.path.isfile(filename):
                print(f'Deleting {filename}')
                os.remove(filename)

            print(f'Saving {len(all_rows)} file names to {filename}')
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(all_rows)
