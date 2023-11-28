"""
Functions to convert raw audio to MFCC data including data augmentation
"""
from enum import Enum
from sonopy import mfcc_spec as sonopy_mfcc_spec
import numpy as np

from wake.parameters import AudioParams
from .spec_augmentation import spec_augment
from .waveform_augmentation import *


class AugmentationType(Enum):
    """
    Enum for the different types of data augmentation
    """
    # Spectrogram augmentation
    SPEC_AUGMENT = 0
    # Waveform augmentation
    NOISE_INJECTION = 1
    CHANGE_SPEED = 2
    CHANGE_PITCH = 3
    # No augmentation
    NONE = 4


def convert_audio(audio: np.ndarray, params: AudioParams, trim_beginning=False, augment=True):
    """
    Takes raw audio and augments the data, outputting the MFCC shaped for input
    into the model
    Args:
        audio: raw audio data
        trim_beginning: if True, trims the beginning of the audio to fit into
            the model
        augment: if True, randomly augments the audio
    Returns:
        MFCC data shaped for input into the model
    """
    # randomly choose an augmentation type
    if augment:
        augmentation_type = np.random.choice(list(AugmentationType))
    else:
        augmentation_type = AugmentationType.NONE
    mfcc = convert_audio_helper(
        audio, trim_beginning, augmentation_type, params)

    return mfcc


def convert_audio_helper(audio: np.ndarray, trim_beginning: bool, augmentation_type: AugmentationType, ap: AudioParams):
    """
    Takes raw audio and augments the data, outputting the MFCC shaped for input
    into the model
    Args:
        audio: raw audio data
        trim_beginning: if True, trims the beginning of the audio to fit into
            the model
        augmentation_type: the type of augmentation to apply
    Returns:
        MFCC data shaped for input into the model
    """

    # If any of the waveform augmentation types
    if augmentation_type == AugmentationType.NOISE_INJECTION:
        audio = noise_injection(audio)
    elif augmentation_type == AugmentationType.CHANGE_SPEED:
        # choose a random speed factor between 0.75 and 1.25 with steps of 0.1
        speed_factor = np.random.choice(np.arange(0.75, 1.25, 0.1))
        audio = change_speed(audio, speed_factor)
    elif augmentation_type == AugmentationType.CHANGE_PITCH:
        pitch_factor = np.random.choice(np.arange(-0.6, 0.5, 0.1))
        audio = change_pitch(audio, pitch_factor)

    # Trim beginning of audio if too long so will fit into single input to model
    if trim_beginning:
        if len(audio) > ap.buffer_samples:
            audio = audio[-ap.buffer_samples:]

    # convert to MFCC
    mfcc = calculate_mfccs(audio, ap)

    # If spectrogram augmentation
    if augmentation_type == AugmentationType.SPEC_AUGMENT:
        mfcc = spec_augment(mfcc)

    # Reshape to fit into model
    mfcc = reshape_mfcc(mfcc, ap)

    if trim_beginning:
        assert (
            len(mfcc) == 1), f"len(mfcc) = {len(mfcc)}\t{audio.shape}\t{augmentation_type}"
    return mfcc


def reshape_mfcc(mfcc: np.ndarray, ap: AudioParams):
    """
    Reshapes the MFCC to fit into the model
    Args:
        mfcc: MFCC data
    Returns:
        MFCC data reshaped to fit into the model
    """
    padded_mfcc = pad_beginning(mfcc, ap)
    input_data = np.reshape(padded_mfcc, (-1, ap.n_features, ap.n_mfcc))
    return input_data


def pad_beginning(mfcc, ap: AudioParams):
    """
    Pads the beginning of the MFCC with zeros so it will fit into the model
    Args:
        mfcc: MFCC data
    Returns:
        MFCC data padded with zeros
    """
    remainder = len(mfcc) % ap.n_features

    if (remainder != 0):
        # append zeros to beginning
        count_zeros = ap.n_features - remainder
        mfcc = np.concatenate([
            np.zeros((count_zeros, mfcc.shape[1])),
            mfcc
        ])
    assert (len(mfcc) % ap.n_features == 0)
    return mfcc


def calculate_mfccs(audio: np.ndarray, ap: AudioParams):
    """
    Converts raw audio to MFCC data
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
