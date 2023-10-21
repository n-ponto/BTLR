
from pyaudio import PyAudio, Stream
from .utils import AudioParams
import time
from tensorflow import keras
import numpy as np
from sonopy import mfcc_spec
import audioop

buffer_samples = 24000
window_samples = 1600
n_features = 29
n_mfcc = 13

sample_rate = 16000
hop_samples = 800

n_fft = 512
n_filt = 20
n_mfcc = 13

# audio_buffer = np.zeros(buffer_samples, dtype=float)
window_audio = np.array([])
listener_mfccs = np.zeros((n_features, n_mfcc))


def update_vectors(stream):
    global window_audio
    global listener_mfccs
    # Convert the buffer to a normalized numpy array
    buffer_audio = np.fromstring(stream, dtype='<i2').astype(
        np.float32, order='C') / 32768.0
    # Add the buffer to the end of the window audio
    # print(buffer_audio.shape, window_audio.shape)
    window_audio = np.concatenate((window_audio, buffer_audio))

    if len(window_audio) >= window_samples:
        # TODO: look at doc of how this function works, I might not have to do these checks
        new_features = mfcc_spec(window_audio, sample_rate, (window_samples, hop_samples),
                                 num_filt=n_filt, fft_size=n_fft, num_coeffs=n_mfcc)
        # Remove the beginning of the window of size len(new_features) * hop_samples
        window_audio = window_audio[len(new_features) * hop_samples:]
        if len(new_features) > len(listener_mfccs):
            # Make sure the new_features are no longer than the MFCCs
            new_features = new_features[-len(listener_mfccs):]
        listener_mfccs = np.concatenate(
            (listener_mfccs[len(new_features):], new_features))
    return listener_mfccs


def load_model():
    model_path = './models/2023-10-21-11.55.47-model.keras'
    return keras.models.load_model(model_path)


def create_stream():
    p = PyAudio()
    stream = p.open(
        rate=AudioParams.sample_rate,
        channels=1,
        format=AudioParams.format,
        input=True,
        frames_per_buffer=AudioParams.chunk_size
    )

    # Wrap the stream read
    # converts pyaudio from reading in terms of samples to in terms of bytes (divide by 2 bc 2 bytes per sample)
    stream.read = lambda x: Stream.read(stream, x // 2, False)

    return p, stream


def listen(stream, model):
    while True:
        data = stream.read(AudioParams.chunk_size)
        prob = get_prediction(data, model)
        rms = audioop.rms(data, 2)
        print(f'rms: {rms}\t\tprob: {prob}', end='\r')

        time.sleep(0.05)


def get_prediction(chunk, model):
    mfccs = update_vectors(chunk)
    raw_output = model.predict(mfccs[np.newaxis], verbose=0)[0][0]
    return raw_output



if __name__ == '__main__':
    try:
        model = load_model()
        p, stream = create_stream()
        listen(stream, model)
    except KeyboardInterrupt:
        print ('ending process')
    finally:
        stream.close()
        p.terminate()