from pyaudio import paInt16


class AudioParams:
    sample_rate = 16000  # samples to record each second
    chunk_size = 1024  # record in chunks of 1024
    channels = 1  # record mono
    format = paInt16  # datatype of

class FileParams:
    data_dir = './data'
    log_dir = './log'
    pos_sample_dir = 'pos'
    neg_sample_dir = 'neg'

    pos_file_name = 'pos'
    neg_file_name = 'neg'

class MycroftParams:
    buffer_samples = 24000
    n_features = 29
    sample_rate = 16000
    window_samples = 1600
    hop_samples = 800
    n_filt = 20
    n_fft = 512
    n_mfcc = 13
    chunk_size = 2048  # Mycroft uses 2048 but wraps stream.read to divide by 2 (i.e. 1024, which is what AI-hacker uses)
    format = paInt16