import librosa
from parameters import MycroftParams as ap
import numpy as np
from data import get_mfcc
import os


def load_noise(path):
    audio, _ = librosa.load(path, sr=ap.sample_rate)
    audio = np.squeeze(audio)
    audio = audio.astype(np.float32)
    mfcc = get_mfcc(audio)
    input = reshape_mfcc(mfcc)
    return input


def reshape_mfcc(mfcc):
    # Reshape to fit into model

    # make sure it is even multiple of size n_features
    remainder = len(mfcc) % ap.n_features

    if (remainder != 0):
        # append zeros to beginning
        count_zeros = ap.n_features - remainder
        mfcc = np.concatenate([
            np.zeros((count_zeros, mfcc.shape[1])),
            mfcc
        ])
    assert (len(mfcc) % ap.n_features == 0)

    # sections = len(mfcc) // ap.n_features
    # input = np.split(mfcc, sections) # shape (n x n_features x n_mfcc)
    input = np.reshape(mfcc, (-1, ap.n_features, ap.n_mfcc))
    return input


def load_noise_dir(directory: str):
    input_parts = []
    files = os.listdir(directory)[:100]
    print(f'loading {len(files)} noise files from {directory}')
    for file in files:
        path = os.path.join(directory, file)
        x = load_noise(path)
        input_parts.append(x)
    inputs = np.concatenate(input_parts)
    outputs = np.zeros((len(inputs)))
    print(f'From {len(files)} files got {len(inputs)} samples.')
    return inputs, outputs


def get_noise():
    noise_dir = "C:\\Users\\noah\\repos\\BTLR\\wake\\data\\cv-corpus-15.0-delta-2023-09-08\\en\\clips"
    return load_noise_dir(noise_dir)
