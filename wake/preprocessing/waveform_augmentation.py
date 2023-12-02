import numpy as np
import librosa


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
    Changes the speed of the audio. A speed_factor of 1.0 will not change the speed.
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def change_pitch(audio, sample_rate: int, pitch_factor=0.4):
    """
    Changes the pitch of the audio. A pitch_factor of 0.0 will not change the pitch.
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)


if __name__ == "__main__":
    print('demo of waveform_augmentation.py')
    import sounddevice as sd
    import time

    # Get the default audio parameters
    import sys
    import os
    path_to_wake = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(path_to_wake)
    from parameters import DEFAULT_AUDIO_PARAMS
    sample_rate = DEFAULT_AUDIO_PARAMS.sample_rate
    print(f'sample_rate = {sample_rate}')

    # load audio
    path = os.path.join(path_to_wake, 'data', 'pos', 'pos-0000.wav')
    assert (os.path.exists(path) and os.path.isfile(
        path)), f'path {path} does not exist'
    audio, _ = librosa.load(path, sr=sample_rate)
    print('loaded audio')
    audio = np.squeeze(audio)
    # print(audio)
    print(f'squeezed audio to shape {audio.shape}')
    audio = audio.astype(np.float32)

    def play(audio_to_play: np.ndarray):
        print('playing audio')
        sd.play(audio_to_play, sample_rate)
        time.sleep(len(audio_to_play) / sample_rate)
        print('done playing audio')
        sd.stop()

    # play audio
    print('original audio')
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
