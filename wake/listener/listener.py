import audioop
import time
import numpy as np
from sonopy import mfcc_spec
from parameters import AudioParams
from listener.activationtrigger import ActivationTrigger
from audio_collection.utils import create_stream
from neuralmodels.modelwrapper import ModelWrapper
import threading

save_index = None


def save_activation(window_audio: np.ndarray, sample_size: int) -> None:
    global save_index
    from audio_collection.utils import save_wav_file
    from audio_collection.utils import get_greatest_index
    import os
    print()
    SAVE_DIR = r".\acivations"
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        print(f"Created directory {SAVE_DIR}")

    # Convert window audio to bytes
    print(f'window_audio.shape = {window_audio.shape}')
    bytes_audio = (window_audio * 32768.0).astype(np.int16).tobytes()
    print(f'bytes_audio.shape = {len(bytes_audio)}')
    print(f'type = {type(bytes_audio)}')
    print(f'len in seconds = {len(bytes_audio)/sample_size/ap.sample_rate}')

    # Get the save index
    if save_index is None:
        save_index = get_greatest_index(SAVE_DIR) + 1

    # Save
    filename = f"{SAVE_DIR}\\activation-{save_index}.wav"
    save_wav_file(filename, sample_size, bytes_audio)
    print(f"Saved activation {filename}")
    save_index += 1


class Listener:
    _model: ModelWrapper
    window_audio: np.ndarray
    input_features: np.ndarray

    def __init__(self, model: ModelWrapper, ap: AudioParams):
        self._model = model
        self.trigger = ActivationTrigger()
        self.window_audio = np.array([])
        self.feature_audio = np.zeros((ap.n_features * ap.hop_samples))
        self.input_features = np.zeros((ap.n_features, ap.n_mfcc))
        self.sample_size = None
        self._stream = None
        self._p = None
        self._listening = False
        self._last_activation_audio = None
        self._ap = ap

    def start_listening(self):
        try:
            # Start the listener thread
            self._listening = True
            thread = threading.Thread(target=self._listen, daemon=True)
            thread.start()

            # Wait for the user to stop the program
            user_input = None
            while user_input != 'q':
                user_input = input('Press q to quit\n')
                if user_input == 'f':
                    save_activation(
                        self._last_activation_audio, self.sample_size)
        except KeyboardInterrupt:
            print('ending process')
        finally:
            # Stop the thread
            print('stopping listener thread...')
            self._listening = False
            time.sleep(0.5)
            thread.join()

            # Clean up the stream
            if self._stream is not None:
                print('closing stream...')
                self._stream.close()
                self._p.terminate()
            print('\n\nDONE')

    def _listen(self):
        self._p, self._stream = create_stream()
        self.sample_size = self._p.get_sample_size(self._ap.format)
        print(f"Sample size = {self.sample_size}")
        while self._listening:
            data = self._stream.read(self._ap.chunk_size, False)
            rms = audioop.rms(data, 2)
            mfccs = self._get_features(data)
            assert (mfccs.shape == (self._ap.n_features, self._ap.n_mfcc)
                    ), f'mfccs.shape = {mfccs.shape}'
            prediction: float = self._model.predict(mfccs)
            triggered = self.trigger.check_trigger(prediction)
            print(f"prediction: {prediction}         ", end='\r')
            # prediction_str = "." * int(prediction * 20)
            if (triggered):
                print("WAKE UP!!!")
                self._last_activation_audio = self.feature_audio.copy()
            # print(f'prediction: {round(prediction, 3):>7}\t|{prediction_str:<20}|\trms:{rms:<10}  ', end='\r')
            # time.sleep(0.05)

    def _get_features(self, buffer: bytes) -> np.ndarray:
        # Convert the buffer to a normalized numpy array
        np_audio = np.fromstring(buffer, dtype='<i2').astype(
            np.float32, order='C') / 32768.0
        self.window_audio = np.concatenate((self.window_audio, np_audio))

        # If there are enough samples, extract new MFCC features
        if len(self.window_audio) >= self._ap.window_samples:
            # Convert audio into MFCC features
            new_features = self._get_mfccs(self.window_audio)

            # The number of audio frames used to create the new_features
            audio_frames_used: int = len(new_features) * self._ap.hop_samples
            new_features_audio: np.ndarray = self.window_audio[:audio_frames_used]

            # Save the audio associated with the new features
            self.feature_audio = np.concatenate(
                (self.feature_audio[audio_frames_used:], new_features_audio))

            # Remove the samples that were used to create the new_features
            self.window_audio = self.window_audio[audio_frames_used:]

            # Make sure the new_features will fit in the model
            if len(new_features) > len(self.input_features):
                new_features = new_features[-len(self.input_features):]

            # Add the new features to the end of the existing features
            self.input_features = np.concatenate(
                (self.input_features[len(new_features):], new_features))

        return self.input_features

    def _get_mfccs(self, window_audio: np.ndarray):
        return mfcc_spec(audio=window_audio,
                         sample_rate=self._ap.sample_rate,
                         window_stride=(self._ap.window_samples, self._ap.hop_samples),
                         num_filt=self._ap.n_filt,
                         fft_size=self._ap.n_fft,
                         num_coeffs=self._ap.n_mfcc)
