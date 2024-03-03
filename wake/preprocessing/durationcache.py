import pickle
import os
import librosa
from wake.parameters import DEFAULT_AUDIO_PARAMS as AP

MIN_FILE_DURATION = ((AP.n_features - 1) * AP.hop_samples +
                     AP.window_samples) / AP.sample_rate


class DurationCache:
    """
    Caches the duration of files so we don't have to keep loading them
    """
    duration_cache: dict  # file_path -> duration
    total_retrieved: int
    count_retrieved_from_cache: int

    def __init__(self, cache_location: str):
        self.duration_cache = {}  # The cache itself (file_path -> duration)
        self.cache_location = cache_location  # The path to the cache file
        self.load_cache()  # Load the cache from disk
        self.total_retrieved = 0
        self.count_retrieved_from_cache = 0

    def load_cache(self):
        """
        Loads the cache from disk
        """
        print('Loading cache...')
        if os.path.isfile(self.cache_location):
            with open(self.cache_location, 'rb') as f:
                self.duration_cache = pickle.load(f)
                print('Done loading existing duration cache.')
        else:
            print('Creating new duration cache')
            self.duration_cache = {}

    def __del__(self):
        self.save()

    def save(self):
        """
        Saves the cache to disk
        """
        print('Saving cache...')
        percent_from_cache = self.count_retrieved_from_cache / \
            self.total_retrieved if self.total_retrieved > 0 else 0
        print(
            f'Durations from cache: {self.count_retrieved_from_cache} / {self.total_retrieved}\t{round(percent_from_cache * 100, 2)}%')
        with open(self.cache_location, 'wb') as f:
            pickle.dump(self.duration_cache, f)

    def get_duration(self, file_path: str):
        """
        Gets the duration of the file in seconds
        Args:
            file_path: The path to the file
        Returns:
            The duration of the file in seconds
        """
        self.total_retrieved += 1
        if file_path in self.duration_cache:
            self.count_retrieved_from_cache += 1
            return self.duration_cache[file_path]

        if file_path.endswith('.wav') or file_path.endswith('.mp3'):
            duration = librosa.get_duration(path=file_path)
        else:
            raise ValueError(f'Cannot get duration of {file_path}')
        duration = max(MIN_FILE_DURATION, duration)
        self.duration_cache[file_path] = duration
        return duration
