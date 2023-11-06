import csv
import numpy as np
import librosa
from audio_conversion import convert_audio
import threading
from pyache import Pyache
import time


def load_files(csv_path: str, augment: bool = True):
    """
    Loads the files from the given CSV file.
    Args:
        csv_path: Path to the CSV file
        augment: Whether to augment the audio
    Returns:
        Tuple of input and output data
    """

    data_loader = DataLoader(csv_path, augment)
    input, output = data_loader.load_data()
    print(f'Loaded {len(input)} samples from {csv_path}')
    return input, output


class DataLoader():
    """
    Loads the data from the CSV file
    """

    NUMBER_THREADS = 8  # Number of threads to use for loading the data

    _pyache: Pyache  # Pyache instance for caching audio files
    _total_files: int  # Total number of files to load
    _augment: bool  # Whether to augment the audio
    _files_to_load = []  # List of files to load
    _threads = []  # List of threads
    _input_parts = []  # List of input data parts
    _output_parts = []  # List of output data parts
    _lock = threading.Lock()  # Lock for accessing the list of files to load

    _input = None  # Input data
    _output = None  # Output data

    def __init__(self, csv_path: str, augment: bool = True):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            self._files_to_load = [(f, bool(int(o))) for f, o in reader]
            print(f'Found {len(self._files_to_load)} files in {csv_path}')
        self._total_files = len(self._files_to_load)
        self._augment = augment
        self._pyache = Pyache('.cache', load_audio_from_file, 'audio_loader')

    def load_data(self):
        """
        Loads the data from the CSV file
        Returns:
            Tuple of input and output data
        """
        if self._input is not None:
            return self._input, self._output
        self._start_time = time.time()
        for _ in range(self.NUMBER_THREADS):
            t = threading.Thread(target=self._thread_function)
            self._threads.append(t)
            t.start()
        for t in self._threads:
            t.join()
        print()
        assert (len(self._input_parts) > 0), 'No data loaded'
        input = np.concatenate(self._input_parts)
        output = np.concatenate(self._output_parts)
        assert (len(input) == len(output)), f'{len(input)} != {len(output)}'
        return input, output

    def _thread_function(self):
        """
        Thread function for loading the data
        """
        while len(self._files_to_load) > 0:
            with self._lock:
                if len(self._files_to_load) % 10 == 0 and len(self._input_parts) > 0:
                    self._print_progress()
                file_path, pos_sample = self._files_to_load.pop()
            audio = self._pyache.load_file(file_path)
            mfcc = convert_audio(audio,
                                 trim_beginning=pos_sample,  # If positive sample, trim the beginning
                                 augment=self._augment)
            corresponding_output = np.ones(
                (len(mfcc))) if pos_sample else np.zeros((len(mfcc)))
            with self._lock:
                self._input_parts.append(mfcc)
                self._output_parts.append(corresponding_output)

    def _print_progress(self):
        """
        Prints the progress of loading the data
        """
        files_completed = len(self._input_parts)
        percent_complete = files_completed/self._total_files
        percent_complete = round(percent_complete * 100, 2)
        elapsed_time = int(time.time() - self._start_time)
        time_per_file = elapsed_time / files_completed
        time_remaining = int(
            time_per_file * (self._total_files - files_completed))
        elapsed_string = f'{elapsed_time//60:>02}:{round(elapsed_time%60):>02}'
        remaining_string = f'{time_remaining//60:>02}:{round(time_remaining%60):>02}'
        print(f'{percent_complete:>2}%\telapsed = {elapsed_string}\testimated remaining = {remaining_string}  ', end='\r')


def load_audio_from_file(file_path: str):
    """
    Loads the audio from the given file.
    Args:
        file_path: Path to the audio file
    Returns:
        Audio data
    """
    # Load audio from file
    audio, sr = librosa.load(file_path, sr=None)
    # Convert to mono
    audio = librosa.to_mono(audio)
    # Convert to float32
    audio = audio.astype(np.float32)
    return audio
