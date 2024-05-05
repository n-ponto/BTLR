from math import ceil
import pyaudio


class FileParams:
    """Holds the parameters for file paths and directories."""

    data_dir = '.\\data'  # Directory containing the audio samples
    log_dir = './log'  # Directory to save the training logs
    pos_sample_name = 'pos'  # Directory containing the positive samples
    neg_sample_name = 'neg'  # Directory containing the negative samples
    noise_sample_name = 'noise'  # Directory containing the noise samples
    default_model_path = 'wake/trained_model.tflite'  # Path to the default model


class AudioParams:
    """Holds the parameters for audio sampling and MFCC calculation."""
    
    """The format of the audio samples."""
    format = pyaudio.paInt16

    """Using mono audio."""
    channels = 1

    """The number of samples returned each time we read from the audio stream."""
    chunk_size: int

    """The number of samples to record each second."""
    sample_rate: int

    """The length of the audio input into the neural network in seconds."""
    features_t: float

    """Length of the MFCC window in seconds."""
    window_t: float

    """Length of the MFCC hop in seconds."""
    hop_t: float

    """Number of MFCCs to get from each window."""
    n_mfcc: int

    """Number of samples to use in each FFT calculation."""
    n_fft: int

    """Number of filters to use in the MFCC calculation."""
    n_filt: int

    def __init__(self,
                 # Parameters for audio recording
                 chunk_size,
                 sample_rate,
                 # Parameter for model input
                 features_t,
                 # Parameters for MFCC calculation
                 window_t,
                 hop_t,
                 n_fft,
                 n_filt,
                 n_mfcc):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.window_t = window_t
        self.hop_t = hop_t
        self.features_t = features_t
        self.n_fft = n_fft
        self.n_filt = n_filt
        self.n_mfcc = n_mfcc

    @property
    def window_samples(self):
        """Samples used to calculate each MFCC."""
        return ceil(self.window_t * self.sample_rate)

    @property
    def hop_samples(self):
        """Samples the window moves between MFCC calculations."""
        return ceil(self.hop_t * self.sample_rate)

    @property
    def feature_samples(self):
        """Samples used to create each input to the neural network."""
        samples = ceil(self.features_t * self.sample_rate)
        # Make sure it's evenly divisible by the hop size
        return (samples // self.hop_samples) * self.hop_samples

    @property
    def n_features(self):
        """
        Based on the parameters for audio sampling and MFCC calculation, this is
        the number of features that will be input into the neural network.

        This is the number of windows that fit inside the features_t with the
        given hop_t.
        """
        return (self.feature_samples - self.window_samples) // self.hop_samples + 1

    @property
    def chunks_per_sec(self):
        """The number of chunks recorded each second."""
        return self.sample_rate / self.chunk_size


DEFAULT_AUDIO_PARAMS = AudioParams(
    chunk_size=1024,
    sample_rate=16000,
    window_t=0.1,
    hop_t=0.05,
    features_t=1.5,
    n_fft=512,
    n_filt=20,
    n_mfcc=13
)
