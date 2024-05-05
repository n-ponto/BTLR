import csv
import numpy as np
import librosa
import threading
import time
import os
from preprocessing.audio_conversion import audio_to_features
from parameters import AudioParams, DEFAULT_AUDIO_PARAMS as AP, FileParams as FP

NUMBER_THREADS = 2  # Number of threads to use for loading the data


class DataLoader():
    """Loads the data from the CSV file."""
    _total_files: int  # Total number of files to load
    _augment: bool  # Whether to augment the audio
    _audioParams: AudioParams  # Parameters for audio conversion

    _files_to_load: list[(str, bool, bool)] = []  # List of files to load
    _input_parts: list[np.ndarray] = []  # List of input data parts
    _output_parts: list[np.ndarray] = []  # List of output data parts
    _threads: list[threading.Thread] = []  # List of threads
    _lock = threading.Lock()  # Lock for accessing the list of files to load
    _start_time: float  # Time when the loading started
    _time_remaining: int = None  # Time remaining to load the data

    def __init__(self, csv_path: str, ap: AudioParams = AP, augment: bool = True):
        """Initializes the data loader.
        Args:
            csv_path: Path to the CSV file
            ap: Audio parameters
            augment: If the audio should be augmented using data augmentation. 
                If True, augments audio if the augment column of the CSV file is 1.
                If False, does not augment anything.
        """
        with open(csv_path, newline='') as f:
            # Columns are: file_path, label, augment
            reader = csv.reader(f)
            self._files_to_load = [(f, bool(int(o)), bool(int(a))) for f, o, a in reader]
            print(f'Found {len(self._files_to_load)} files in {csv_path}')
        self._total_files = len(self._files_to_load)
        self._augment = augment
        self._audioParams = ap

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data from the CSV file. Uses the file path from the CSV
        file to load the audio, perform data augmentation, convert to MFCC,
        and return the input and output data.
        Returns:
            Tuple of input and output data
        """
        # Create multiple threads to load audio from disk
        self._start_time = time.time()
        for _ in range(NUMBER_THREADS):
            t = threading.Thread(target=self._thread_function)
            self._threads.append(t)
            t.start()

        # Wait for the threads to finish
        for t in self._threads:
            t.join()

        print()
        if len(self._input_parts) <= 0:
            print('WARNING: No data loaded. Check the CSV file path and the audio files.')
            exit(-1)

        # Join all the data loaded into one long array
        input = np.concatenate(self._input_parts)
        output = np.concatenate(self._output_parts)
        assert (len(input) == len(output)), f'Expected same length {len(input)} != {len(output)}'
        return input, output

    def _thread_function(self):
        """Thread function for loading the data."""
        while len(self._files_to_load) > 0:
            with self._lock:
                # Print output every 10 files loaded
                if len(self._files_to_load) % 10 == 0 and len(self._input_parts) > 0:
                    self._print_progress()
                file_path, label, augment = self._files_to_load.pop()
            augment = augment and self._augment  # Augment if CSV column is 1 and augment is True
            audio = self._load_audio_from_file(file_path)
            if audio is None:
                # Skip empty audio files
                continue
            mfcc = audio_to_features(audio,
                                     self._audioParams,
                                     trim_beginning=label,  # If positive sample, trim the beginning
                                     augment=augment)

            corresponding_output = np.ones((len(mfcc))) if label else np.zeros((len(mfcc)))

            with self._lock:
                self._input_parts.append(mfcc)
                self._output_parts.append(corresponding_output)

    def _load_audio_from_file(self, file_path: str) -> np.ndarray:
        """Loads the audio from the given file.
        Args:
            file_path: Path to the audio file
        Returns:
            Audio data as an np.ndarray
        """
        file_path = os.path.join(FP.data_dir, file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        # Load audio from file
        audio, _ = librosa.load(file_path, sr=self._audioParams.sample_rate)
        # Convert to mono
        audio = librosa.to_mono(audio)
        # Convert to float32
        audio = audio.astype(np.float32)
        if len(audio) <= 0:
            print("WARNING: Empty audio file", file_path)
            return None
        return audio

    def _print_progress(self):
        """Prints the progress of loading the data."""
        files_completed = len(self._input_parts)
        percent_complete = files_completed/self._total_files
        percent_complete = round(percent_complete * 100, 2)
        elapsed_time = int(time.time() - self._start_time)
        time_per_file = elapsed_time / files_completed
        time_remaining = int(time_per_file * (self._total_files - files_completed))
        if self._time_remaining is not None:
            # Calculate moving average of time remaining
            self._time_remaining = int((self._time_remaining * 0.9) + (time_remaining * 0.1))
        else:
            self._time_remaining = time_remaining
        elapsed_string = f'{elapsed_time//60:>02}:{round(elapsed_time%60):>02}'
        remaining_string = f'{self._time_remaining//60:>02}:{round(self._time_remaining%60):>02}'
        print(f'{percent_complete:>2}%\telapsed = {elapsed_string}\testimated remaining = {remaining_string}  ', end='\r')
