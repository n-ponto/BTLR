import csv
import numpy as np
import librosa
from preprocessing.audio_conversion import convert_audio
import threading
from pyache import Pyache
import time

NUMBER_THREADS = 4  # Number of threads to use for loading the data


class DataLoader():
    """
    Loads the data from the CSV file
    """
    _pyache: Pyache  # Pyache instance for caching audio files
    _total_files: int  # Total number of files to load
    _augment: bool  # Whether to augment the audio
    _files_to_load: list[(str, bool, bool)] = []  # List of files to load
    _threads: list[threading.Thread] = []  # List of threads
    _input_parts: list[np.ndarray] = []  # List of input data parts
    _output_parts: list[np.ndarray] = []  # List of output data parts
    _lock = threading.Lock()  # Lock for accessing the list of files to load

    _input = None  # Input data
    _output = None  # Output data

    def __init__(self, csv_path: str, augment: bool = True):
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            self._files_to_load = [(f, bool(int(o)), bool(int(a)))
                                   for f, o, a in reader]
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

        # If already loaded, return the cached data
        if self._input is not None:
            return self._input, self._output

        # Create multiple threads to load audio from disk
        self._start_time = time.time()
        for _ in range(NUMBER_THREADS):
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
                file_path, label, augment = self._files_to_load.pop()
            augment = augment and self._augment
            audio = self._pyache.load_file(file_path)
            mfcc = convert_audio(audio,
                                 trim_beginning=label,  # If positive sample, trim the beginning
                                 augment=augment)

            if label:
                corresponding_output = np.ones((len(mfcc)))
            else:
                corresponding_output = np.zeros((len(mfcc)))

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
