from pyaudio import paInt16


class AudioParams:
    sample_rate = 16000  # samples to record each second
    chunk_size = 1024  # record in chunks of 1024
    channels = 1  # record mono
    format = paInt16  # datatype of

class FileParams:
    default_data_dir = './data'
    pos_sample_dir = 'wake-word'
    neg_sample_dir = 'not-wake-word'

    pos_file_name = 'pos'
    neg_file_name = 'neg'

class MycroftParams:
    sample_rate = 16000
    window_samples = 1600
    hop_samples = 800
    n_filt = 20
    n_fft = 512
    n_mfcc = 13