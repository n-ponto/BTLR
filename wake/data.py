"""
Code for loading in the collected data in a format for training
"""
from sonopy import mfcc_spec
from parameters import AudioParams, MycroftParams
import wavio
import wave
import numpy as np
import os

def get_mfcc_youtube(audio):
    return mfcc_spec(
        audio=audio,
        sample_rate=AudioParams.sample_rate,
        window_stride=(400,200),
        fft_size=400,
        num_filt=40,
        num_coeffs=40
    )

def get_mfcc(audio):
    return mfcc_spec(
        audio=audio,
        sample_rate=MycroftParams.sample_rate,
        window_stride=(MycroftParams.window_samples, MycroftParams.hop_samples),
        num_filt=MycroftParams.n_filt,
        fft_size=MycroftParams.n_fft,
        num_coeffs=MycroftParams.n_mfcc
    )

max_samples = 24000
n_features = 29

def vectorize(audio):
    # Remove beginning of audio if too long
    if len(audio) > max_samples:
        audio = audio[-max_samples:]
    features = get_mfcc(audio)
    # If not enough timesteps, append zeros to beginning
    if len(features) < n_features:
        features = np.concatenate([
            np.zeros((n_features - len(features), features.shape[1])),
            features
        ])
    # If too many timesteps, only keep the end
    if len(features) > n_features:
        features = features[-n_features:]
    return features

def load_audio(path: str):
    wav = wavio.read(path)
    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)    

def load_file(path: str):
    assert(os.path.isfile(path)), path
    audio = load_audio(path)
    return vectorize(audio)

def load_dir(path: str, pos: bool):
    # v = load_file(os.path.join(path, os.listdir(path)[0]))
    files = os.listdir(path)
    # a = load_audio(os.path.join(path, files[1]))
    # v = vectorize(a)

    input_parts = [load_file(os.path.join(path, f)) for f in files]
    output_parts = np.array([[pos] for _ in input_parts])

    inputs = np.stack(input_parts, axis=0)  # TODO: figure out how to concat these arrays, maybe look at youtube github
    outputs = np.concatenate(output_parts, axis=0)
    return inputs, outputs

def load_data(path: str, pos_dir: str):
    dirs = os.listdir(path)
    input_parts = []
    output_parts = []
    for dir in dirs[:2]:
        full_path = os.path.join(path, dir)
        print(full_path)
        data = load_dir(full_path, dir == pos_dir)
        print(f'Got {len(data[0])} samples from {full_path}')
        input_parts.append(data[0])
        output_parts.append(data[1])
    input = np.concatenate(input_parts)
    output = np.concatenate(output_parts)
    return input, output
                

def get_data():
    data_path = "C:\\Users\\noah\\repos\\mycroft-precise\\hey-computer"
    pos_dir = "wake-word"

    input, output = load_data(data_path, pos_dir)
    
    return input, output