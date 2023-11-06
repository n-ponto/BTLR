"""
Functions to convert raw audio to MFCC data including data augmentation
"""
from enum import Enum
import librosa
from parameters import MycroftParams as ap
from sonopy import mfcc_spec as sonopy_mfcc_spec
import numpy as np

# The maximum number of samples that fit into a single input to the model
MAX_SAMPLES = (ap.n_features - 1) * ap.hop_samples + ap.window_samples
assert (MAX_SAMPLES == 24000 and MAX_SAMPLES == ap.sample_rate * 1.5)


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


def convert_audio(audio: np.ndarray, trim_beginning=False, augment=True):
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

    mfcc = convert_audio_helper(audio, trim_beginning, augmentation_type)

    return mfcc


def convert_audio_helper(audio: np.ndarray, trim_beginning: bool, augmentation_type: AugmentationType):
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
        if len(audio) > MAX_SAMPLES:
            audio = audio[-MAX_SAMPLES:]

    # convert to MFCC
    mfcc = get_mfcc(audio)

    # If spectrogram augmentation
    if augmentation_type == AugmentationType.SPEC_AUGMENT:
        mfcc = spec_augment(mfcc)

    # Reshape to fit into model
    mfcc = reshape_mfcc(mfcc)

    if trim_beginning:
        assert (len(mfcc) == 1), f"len(mfcc) = {len(mfcc)}\t{audio.shape}\t{augmentation_type}"
    return mfcc


def get_mfcc(audio: np.ndarray):
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


# region Spectrogram augmentation


def spec_augment(mfcc: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """
    Applies spectrogram augmentation to the MFCC
    Args:
        mfcc: MFCC data
        num_mask: number of masks to apply
        freq_masking_max_percentage: maximum percentage of frequencies to mask
        time_masking_max_percentage: maximum percentage of time to mask
    Returns:
        MFCC data with spectrogram augmentation applied
    """
    assert (len(mfcc.shape) == 2), f'Expected 2D array, got {mfcc.shape}'
    mfcc = mfcc.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = mfcc.shape
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        mfcc[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(
            low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        mfcc[t0:t0 + num_frames_to_mask, :] = 0

    return mfcc

# endregion
# region Waveform augmentation


def noise_injection(audio: np.ndarray, noise_factor=0.005):
    """
    Adds random noise to the audio
    Args:
        audio: raw audio data
        noise_factor: strength of the noise
    Returns:
        audio with added noise
    """
    noise = np.random.randn(len(audio))
    data = audio + noise_factor * noise
    return data


def change_speed(audio: np.ndarray, speed_factor=0.9):
    """
    Changes the speed of the audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def change_pitch(audio, pitch_factor=0.4):
    """
    Changes the pitch of the audio
    """
    return librosa.effects.pitch_shift(audio, sr=ap.sample_rate, n_steps=pitch_factor)


def reshape_mfcc(mfcc: np.ndarray):
    """
    Reshapes the MFCC to fit into the model
    Args:
        mfcc: MFCC data
    Returns:
        MFCC data reshaped to fit into the model
    """
    padded_mfcc = pad_beginning(mfcc)
    input_data = np.reshape(padded_mfcc, (-1, ap.n_features, ap.n_mfcc))
    # if (clip_beginning): assert(len(input_data) == 1)
    return input_data


def pad_beginning(mfcc):
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
# endregion
# region Demo


if __name__ == "__main__":
    print('demo of DataAugmentor.py')
    import sounddevice as sd
    import time
    print('done importing sounddevice and time')
    path = "C:\\Users\\noah\\repos\\BTLR\\wake\\data\\pos\\pos-00.wav"
    audio, _ = librosa.load(path, sr=ap.sample_rate)
    audio = np.squeeze(audio)
    # print(audio)
    print(f'squeezed audio to shape {audio.shape}')
    audio = audio.astype(np.float32)

    def play(audio_to_play: np.ndarray):
        print('playing audio')
        sd.play(audio_to_play, ap.sample_rate)
        time.sleep(len(audio_to_play) / ap.sample_rate)
        print('done playing audio')
        sd.stop()

    # play audio
    print('plain audio')
    play(audio)

    # convert noise_injection
    print('noise_injection')
    noise_injection_audio = noise_injection(audio)
    play(noise_injection_audio)

    # convert change_speed [0.75, ]
    print('change_speed')
    change_speed_audio = change_speed(audio)
    play(change_speed_audio)

    # convert change_pitch
    print('change_pitch')
    change_pitch_audio = change_pitch(audio)
    play(change_pitch_audio)
# endregion
