"""
Functions to convert raw audio to MFCC data including data augmentation
"""
from enum import Enum
from sonopy import mfcc_spec as sonopy_mfcc_spec
import numpy as np

from parameters import AudioParams
from .spec_augmentation import spec_augment
from .waveform_augmentation import *

# Possible values for choosing how to randomly factor speed and pitch
SPEED_OPTIONS = np.arange(0.75, 1.25, 0.1)
PITCH_OPTIONS = np.arange(-0.6, 0.5, 0.1)


class AugmentationType(Enum):
    """Enum for the different types of data augmentation"""

    # Spectrogram augmentation
    SPEC_AUGMENT = 0
    # Waveform augmentation
    NOISE_INJECTION = 1
    CHANGE_SPEED = 2
    CHANGE_PITCH = 3
    # No augmentation
    NONE = 4


def audio_to_features(audio: np.ndarray, audio_params: AudioParams, trim_beginning=False, augment=True):
    """Takes raw audio and augments the data, outputting the MFCC shaped for input
    into the model
    Args:
        audio: raw audio data
        trim_beginning: if True, trims the beginning of the audio to fit into
            the model (use for pos or neg samples, not noise or downloaded)
        augment: if True, randomly augments the audio
    Returns:
        MFCC data shaped for input into the model
    """
    # Randomly choose an augmentation type
    augmentation_type = np.random.choice(list(AugmentationType)) if augment else AugmentationType.NONE
    
    # Apply waveform augmentation
    if augmentation_type == AugmentationType.NOISE_INJECTION:
        audio = noise_injection(audio)
    elif augmentation_type == AugmentationType.CHANGE_SPEED:
        # choose a random speed factor between 0.75 and 1.25 with steps of 0.1
        speed_factor = np.random.choice(SPEED_OPTIONS)
        audio = change_speed(audio, speed_factor)
    elif augmentation_type == AugmentationType.CHANGE_PITCH:
        pitch_factor = np.random.choice(PITCH_OPTIONS)
        audio = change_pitch(audio, audio_params.sample_rate, pitch_factor)

    # Trim beginning of audio if too long so will fit into single input to model
    if trim_beginning:
        if len(audio) > audio_params.feature_samples:
            audio = audio[-audio_params.feature_samples:]

    # convert to MFCC
    mfcc = calculate_mfcc(audio, audio_params)

    # If spectrogram augmentation
    if augmentation_type == AugmentationType.SPEC_AUGMENT:
        mfcc = spec_augment(mfcc)

    # Reshape to fit into model
    mfcc = reshape_mfcc(mfcc, audio_params)

    # Make sure that if we set trim_beginning to True, we only have one input to the model (i.e. training sample)
    if trim_beginning:
        assert (len(mfcc) == 1), f"len(mfcc) = {len(mfcc)}\t{audio.shape}\t{augmentation_type}"
    return mfcc


def reshape_mfcc(mfcc: np.ndarray, ap: AudioParams):
    """Reshapes the MFCC to fit into the model.

    For example, if the mfcc is 94x13 and the model takes 10x13, this will 
    reshape the data to be 10x10x13 for (10 inputs of 10x13 data into the model)

    Args:
        mfcc: MFCC data
        ap: Audio parameters
    Returns:
        MFCC data reshaped to fit into the model
    """
    padded_mfcc = pad_beginning(mfcc, ap)
    input_data = np.reshape(padded_mfcc, (-1, ap.n_features, ap.n_mfcc))
    return input_data


def pad_beginning(mfcc, ap: AudioParams):
    """Pads the beginning of the MFCC with zeros so it evenly fits into the model

    For example, if the mfcc is 94x13 and the model takes 10x13, this will add
    6 rows of zeros to the beginning (making it 100x13).

    Args:
        mfcc: MFCC data
        ap: Audio parameters
    Returns:
        MFCC data padded with zeros
    """
    remainder = len(mfcc) % ap.n_features

    if (remainder != 0):
        # append zeros to beginning
        count_zeros = ap.n_features - remainder
        mfcc = np.concatenate([np.zeros((count_zeros, mfcc.shape[1])), mfcc])
    assert (len(mfcc) % ap.n_features == 0)
    return mfcc


def calculate_mfcc(audio: np.ndarray, ap: AudioParams):
    """Converts raw audio to MFCC data
    Args:
        audio: raw audio data
    Returns:
        MFCC data
    """
    return sonopy_mfcc_spec(
        audio=audio,
        sample_rate=ap.sample_rate,
        window_stride=(ap.window_samples, ap.hop_samples),
        num_filt=ap.n_filt,
        fft_size=ap.n_fft,
        num_coeffs=ap.n_mfcc
    )
