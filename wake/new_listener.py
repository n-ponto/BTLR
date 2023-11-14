
from pyaudio import PyAudio, Stream
import time
from tensorflow import keras
import numpy as np
from sonopy import mfcc_spec
import audioop
from parameters import MycroftParams as ap


window_audio = np.array([])
listener_mfccs = np.zeros((ap.n_features, ap.n_mfcc))
activation = 0
    
def check_trigger(prob, sensitivity=0.3, trigger_level=3, activation_delay=8):
    global activation
    chunk_activated = prob > 1.0 - sensitivity

    if chunk_activated or activation < 0:
        activation += 1
        has_activated = activation > trigger_level
        if has_activated or chunk_activated and activation < 0:
            # Wait for activation_delay chunks after trigger without any activations
            # before counting new activations
            activation = -activation_delay

        if has_activated:
            return True
    elif activation > 0:
        activation -= 1
    return False


def get_features(stream):
    global window_audio
    global listener_mfccs
    # Convert the buffer to a normalized numpy array
    buffer_audio = np.fromstring(stream, dtype='<i2').astype(
        np.float32, order='C') / 32768.0
    window_audio = np.concatenate((window_audio, buffer_audio))

    if len(window_audio) >= ap.window_samples:
        new_features = mfcc_spec(window_audio, ap.sample_rate, (ap.window_samples, ap.hop_samples),
                                 num_filt=ap.n_filt, fft_size=ap.n_fft, num_coeffs=ap.n_mfcc)
        # Remove the beginning of the window of size len(new_features) * hop_samples
        window_audio = window_audio[len(new_features) * ap.hop_samples:]
        if len(new_features) > len(listener_mfccs):
            # Make sure the new_features are no longer than the MFCCs
            new_features = new_features[-len(listener_mfccs):]
        listener_mfccs = np.concatenate(
            (listener_mfccs[len(new_features):], new_features))
    return listener_mfccs


def load_model():
    model_path = './checkpoints/model3'
    return keras.models.load_model(model_path)

def create_stream():
    p = PyAudio()
    stream = p.open(
        rate=ap.sample_rate,
        channels=1,
        format=ap.format,
        input=True,
        frames_per_buffer=ap.chunk_size
    )

    # Wrap the stream read
    # converts pyaudio from reading in terms of samples to in terms of bytes (divide by 2 bc 2 bytes per sample)
    stream.read = lambda x: Stream.read(stream, x // 2, False)

    return p, stream

def loop(model):
    try:
        p, stream = create_stream()
        while True:
            data = stream.read(ap.chunk_size)
            rms = audioop.rms(data, 2)
            mfccs = get_features(data)
            assert(mfccs.shape == (ap.n_features, ap.n_mfcc)), f'mfccs.shape = {mfccs.shape}'
            prediction = model.predict(mfccs[np.newaxis], verbose=0)[0][0]
            triggered = check_trigger(prediction)
            prediction_str = "." * int(prediction * 20)
            print(f"prediction: {prediction}         ", end='\r')
            if (triggered):
                print("WAKE UP!!!")
            # print(f'prediction: {round(prediction, 3):>7}\t|{prediction_str:<20}|\trms:{rms:<10}  ', end='\r')
            # time.sleep(0.05)
    except KeyboardInterrupt:
        print ('ending process')
    finally:
        stream.close()
        p.terminate()
        print('done')

if __name__ == '__main__':
    model = load_model()
    # model = None
    loop(model)