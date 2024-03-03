"""Tool to collect samples for the wake word detector."""
import audioop
from enum import Enum
import os
import threading
from . import utils

THRESHOLD = 400  # How loud the audio needs to be to trigger activation
ALLOWED_SILENCE_SECS = 0.3  # allowed time of silence in seconds to remain activated
MAX_SAMPLE_LEN_SEC = 1.5  # maximum length of a single sample in seconds
NOISE_FILE_LENGTH_SECONDS = 15  # maximum length of a single noise sample in seconds


class CollectionMode (Enum):
    """Different user input modes for collecting data."""
    ENTER = "enter"
    THRESH = "threshold"
    NOISE = "noise"


class SampleCollector:
    """Class used to collect audio samples from the user."""

    lock = threading.Lock()
    thread: threading.Thread = None
    thread_continue: bool = False

    def __init__(self, audio_params, file_params, mode: CollectionMode | str, negative: bool = False, overwrite: bool = False):
        """Initializes the sample collector.
        Args:
            audio_params: `parameters.AudioParams` the audio parameters
            file_params: `parameters.FileParams` the file parameters
            mode: the mode to collect samples in
            negative: if the samples are negative (default False)
            overwrite: if we should overwrite files in the save directory
        """
        if type(mode) == str:
            mode = mode.lower()
            modes = [m.value for m in CollectionMode]
            if mode not in modes:
                raise ValueError(f"Invalid mode {mode}, must be in {modes}.")
            mode = CollectionMode(mode)

        self.mode = CollectionMode(mode)
        self.audio_params = audio_params
        self.file_params = file_params
        # When collecting noise, always negative samples
        self.negative = True if self.mode == CollectionMode.NOISE else negative
        # Get file save info
        self.save_directory = self._get_directory()
        self.file_prefix = self._get_file_prefix()
        self.save_index = self._get_file_index(overwrite)
        self.sample_size = None
        self.stream = None

    def run(self):
        """Runs the sample collection process."""
        p, stream = utils.create_stream(self.audio_params)
        self.stream = stream
        self.sample_size = p.get_sample_size(self.audio_params.format)

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

    def _get_file_prefix(self) -> str:
        """Gets the prefix to use for saving files."""
        if self.mode == CollectionMode.NOISE:
            return 'noise'
        return self.file_params.neg_file_name if self.negative else self.file_params.pos_file_name

    def _get_directory(self) -> str:
        """Gets the directory to save samples in. If it doesn't exist, creates it."""
        if not os.path.isdir(self.file_params.data_dir):
            os.mkdir(self.file_params.data_dir)

        if self.mode == CollectionMode.NOISE:
            subdir = 'noise'
        else:
            subdir = self.file_params.neg_sample_dir if self.negative else self.file_params.pos_sample_dir
        fulldir = os.path.join(self.file_params.data_dir, subdir)

        if not os.path.isdir(fulldir):
            os.mkdir(fulldir)

        print(f'saving samples to {os.path.abspath(fulldir)}')
        return fulldir

    def _get_file_index(self, overwrite) -> int:
        """Gets the index to start saving files.
        Args:
            overwrite: if we should overwrite files in the save directory
        Returns:
            0 if `overwrite` is True, otherwise returns the highest index in the
            save directory + 1.
        """
        if overwrite:
            return 0
        index = utils.get_greatest_index(self.save_directory) + 1
        print(f'starting from index {index}')
        return index

    def read_stream(self) -> bytes:
        """Reads a chunk of audio data from the stream."""
        return self.stream.read(self.audio_params.chunk_size, False)

    def collect_wakeword_threshold(self, rms_threshold) -> None:
        """Saves audio when volume exceeds the given threshold.
        Args:
            rms_threshold: the threshold to activate recording
        """
        allowed_silence_frames = int(
            self.audio_params.sample_rate/self.audio_params.chunk_size*ALLOWED_SILENCE_SECS)
        silence_frames = 0
        collecting = False
        frames = []
        i = 0
        print('Saves samples when volume exceeds the given threshold.')
        print('Press ctrl-c to stop...')
        while True:
            data = self.read_stream()
            rms = audioop.rms(data, 2)
            print(str(rms) + '   \t', end='')
            if collecting:
                if rms <= rms_threshold:  # Silence during collection
                    if silence_frames > allowed_silence_frames:  # Exceeded allowed silence time, save the sample
                        # Save the sample and reset
                        print('saving   \t', end='')
                        self.save_data(b''.join(frames))
                        i += 1
                        collecting = False
                        frames = []
                        silence_frames = 0
                    else:  # Still within allowed silence time
                        silence_frames += 1  # Record that it was silent
                else:  # If there was a sound, reset silence frames
                    silence_frames = 0
            else:  # Not collecting
                if rms > rms_threshold:
                    # New sound activates collection
                    print('activated\t', end='')
                    collecting = True
                    frames.append(data)

            print('', end='\r')
            if collecting:
                frames.append(data)
            else:
                # Always keep one prefix frame before noise starts
                frames = [data]

    def collect_wakeword_enter(self) -> None:
        """Saves audio when the user presses enter."""
        allowed_chunks = int(MAX_SAMPLE_LEN_SEC *
                             self.audio_params.sample_rate/self.audio_params.chunk_size)

        def helper(queue: list):
            """Adds audio data to the queue"""
            while self.thread_continue and len(queue) < allowed_chunks:
                data = data = self.read_stream()
                with self.lock:
                    queue.append(data)

        frames = []
        input('Press enter to start recording...')
        self.thread = threading.Thread(
            target=helper, args=(frames,))
        self.thread_continue = True
        self.thread.start()
        while True:
            input('Press enter to save sample...')
            with self.lock:
                # Keep most recent audio
                data = b''.join(frames[-allowed_chunks:])
                frames.clear()
            print(f'saving {self.file_prefix} file {self.save_index}')
            self.save_data(data)

    def collect_noise(self) -> None:
        """Collects longer samples of noise to use as negative training samples."""
        chunks_per_file = self.audio_params.sample_rate//self.audio_params.chunk_size * \
            NOISE_FILE_LENGTH_SECONDS
        frames = []
        try:
            while True:
                data = data = self.read_stream()
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

    def save_data(self, data):
        """Saves the given audio data to a wav file."""
        filename = f'{self.file_prefix}-{self.save_index:04d}.wav'
        filename = os.path.join(self.save_directory, filename)
        self.save_index += 1
        utils.save_wav_file(filename, self.sample_size,
                            self.audio_params.sample_rate, data)
