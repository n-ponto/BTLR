"""Tool to collect samples for the wake word detector."""
import audioop
from enum import Enum
import os
import threading
import time
from . import utils

THRESHOLD = 600  # How loud the audio needs to be to trigger activation
ALLOWED_SILENCE_SEC = 0.3  # allowed time of silence in seconds to remain activated
MIN_SAMPLE_LEN_SEC = 0.5  # minimum length of a single sample in seconds
NOISE_FILE_LEN_SEC = 15  # maximum length of a single noise sample in seconds


class CollectionMode (Enum):
    """Different user input modes for collecting data."""
    ENTER = "enter"
    THRESH = "threshold"
    NOISE = "noise"


COLLECTION_MODES: list[str] = [m.value for m in CollectionMode]  # List of allowed modes


class SampleCollector:
    """Class used to collect audio samples from the user."""

    lock = threading.Lock()
    thread: threading.Thread = None
    thread_continue: bool = False
    stream = None   # The audio stream `pyaudio.Stream`
    sample_size = None  # Size of each sample in bytes

    def __init__(self, audio_params, file_params, mode, negative: bool = False):
        """Initializes the sample collector.
        Args:
            audio_params: `parameters.AudioParams` the audio parameters
            file_params: `parameters.FileParams` the file parameters
            mode: `CollectionMode` or `str` the mode to collect samples in
            negative: if the samples are negative (default False)
        """
        if type(mode) == str:
            mode = mode.lower()
            if mode not in COLLECTION_MODES:
                raise ValueError(f"Invalid mode {mode}, must be in {COLLECTION_MODES}.")
            mode = CollectionMode(mode)

        self.mode = CollectionMode(mode)
        self.ap = audio_params
        self.file_params = file_params
        # When collecting noise, always negative samples
        self.negative = True if self.mode == CollectionMode.NOISE else negative
        # Get file save info
        self.file_prefix = self._get_file_prefix()
        self.save_directory = self._get_directory()
        self.save_index = utils.get_greatest_index(self.save_directory) + 1

    def _get_file_prefix(self) -> str:
        """Gets the prefix to use for saving files."""
        if self.mode == CollectionMode.NOISE:
            return self.file_params.noise_sample_name
        return self.file_params.neg_sample_name if self.negative else self.file_params.pos_sample_name

    def _get_directory(self) -> str:
        """Gets the directory to save samples in. If it doesn't exist, creates it."""
        if not os.path.isdir(self.file_params.data_dir):
            os.mkdir(self.file_params.data_dir)
            print('Created directory:', self.file_params.data_dir)

        if self.mode == CollectionMode.NOISE:
            subdir = self.file_params.noise_sample_name
        else:
            subdir = self.file_params.neg_sample_name if self.negative else self.file_params.pos_sample_name
        fulldir = os.path.join(self.file_params.data_dir, subdir)

        if not os.path.isdir(fulldir):
            os.mkdir(fulldir)
            print('Created directory:', fulldir)

        print(f'Saving samples to {os.path.abspath(fulldir)}')
        return fulldir

    def read_stream(self) -> bytes:
        """Reads a chunk of audio data from the stream."""
        return self.stream.read(self.ap.chunk_size, False)

    def save_data(self, data):
        """Saves the given audio data to a wav file."""
        if len(data) < self.sample_size * self.ap.sample_rate * MIN_SAMPLE_LEN_SEC:
            return  # Ignore samples that are too short
        filename = f'{self.file_prefix}-{self.save_index:04d}.wav'
        filename = os.path.join(self.save_directory, filename)
        self.save_index += 1
        utils.save_wav_file(filename, self.sample_size, self.ap.sample_rate, data)

    def collect_wakeword_threshold(self, rms_threshold: int) -> None:
        """Saves audio when volume exceeds the given threshold.
        Args:
            rms_threshold: the threshold to activate recording
        """
        allowed_silence_frames = int(ALLOWED_SILENCE_SEC * self.ap.chunks_per_sec)
        silence_frames = 0
        collecting = False
        frames = []
        print('Saves samples when volume exceeds the given threshold.')
        print('Press ctrl-c to stop...')
        time.sleep(1)
        self.read_stream()  # Clear the buffer

        while True:
            data = self.read_stream()
            volume = audioop.rms(data, 2)
            print(str(volume) + '   \t', end='')
            if collecting:
                if volume <= rms_threshold:  # Silence during collection
                    silence_frames += 1  # Record that it was silent
                    if silence_frames > allowed_silence_frames:  # Exceeded allowed silence time, save the sample
                        # Save the sample and reset
                        print('saving   \t', end='')
                        data = b''.join(frames)
                        self.save_data(data)
                        collecting = False
                        frames = []
                        silence_frames = 0
                else:  # If there was a sound, reset silence frames
                    silence_frames = 0
            elif volume > rms_threshold:  # Not collecting and there's sound
                # New sound activates collection
                print('activated\t', end='')
                collecting = True

            print('', end='\r')
            if collecting:
                frames.append(data)
            else:
                # Always keep one prefix frame before noise starts
                frames = [data]

    def collect_wakeword_enter(self) -> None:
        """Saves audio when the user presses enter."""
        allowed_chunks = int(self.ap.features_t * self.ap.chunks_per_sec)

        def helper(frames: list):
            """Adds audio data to the list."""
            while self.thread_continue:
                data = self.read_stream()
                with self.lock:
                    frames.append(data)
                    # If frames gets too long, remove the oldest frames
                    if len(frames) > allowed_chunks * 4:
                        frames = frames[-allowed_chunks:]

        frames = []
        input('Press enter to start recording...')
        self.thread = threading.Thread(target=helper, args=(frames,))
        self.thread_continue = True
        self.thread.start()  # Start the thread to populate frames
        while True:
            input('Press enter to save sample...')
            with self.lock:
                # Only keep most recent audio within the allowed length
                data = b''.join(frames[-allowed_chunks:])
                frames.clear()
            print(f'saving {self.file_prefix} file {self.save_index}')
            self.save_data(data)

    def collect_noise(self) -> None:
        """Collects longer samples of noise to use as negative training samples."""
        chunks_per_file = int(NOISE_FILE_LEN_SEC * self.ap.chunks_per_sec)
        frames = []
        try:
            while True:
                data = self.read_stream()
                frames.append(data)
                if len(frames) >= chunks_per_file:
                    joined_frames = b''.join(frames)
                    print(f'saving noise sample {self.save_index}')
                    self.save_data(joined_frames)
                    frames = []
        except KeyboardInterrupt:
            print('interrupted...')
        finally:
            if len(frames) > 0:
                joined_frames = b''.join(frames)
                self.save_data(joined_frames)

    def run(self):
        """Runs the sample collection process."""
        p, stream = utils.create_stream(self.ap)
        self.stream = stream
        self.sample_size = p.get_sample_size(self.ap.format)

        try:
            if self.negative:
                print("Please say words OTHER than the wake word...")
            else:
                print("Please say the wake word...")

            if self.mode == CollectionMode.ENTER:
                print("Press enter to save sample...")
                self.collect_wakeword_enter()
            elif self.mode == CollectionMode.THRESH:
                print("Saving samples automatically...")
                self.collect_wakeword_threshold(THRESHOLD)
            elif self.mode == CollectionMode.NOISE:
                print("Saving noise samples...")
                self.collect_noise()
        except KeyboardInterrupt:
            pass
        finally:
            print('\n\nending process...')
            self.thread_continue = False
            if self.thread:
                self.thread.join()
            if self.stream:
                stream.stop_stream()
                stream.close()
