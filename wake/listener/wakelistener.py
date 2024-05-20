import numpy as np
from sonopy import mfcc_spec

from wake.parameters import AudioParams, FileParams, DEFAULT_AUDIO_PARAMS
from wake.modelwrapper import ModelWrapper, get_model_wrapper
from .activationtrigger import ActivationTrigger


class WakeListener:
    audio_buffer: np.ndarray  # The audio data that is being processed
    input_features: np.ndarray  # Feature matrix used for predictions

    _model: ModelWrapper  # The model to use to make predictions on the audio

    def __init__(self,
                 model: ModelWrapper = None,
                 ap: AudioParams = DEFAULT_AUDIO_PARAMS):
        self.trigger = ActivationTrigger()
        # Initialize the audio window to empty
        self.audio_buffer = np.array([])
        # Initialize the feature matrix to zeros (as if it was silent)
        self.input_features = np.zeros((ap.n_features, ap.n_mfcc))
        # Initialize the audio associated with the features to zeros (aka silence)
        self.feature_audio = np.zeros((ap.n_features * ap.hop_samples))

        self._last_activation_audio = None
        self._ap = ap
        if model is not None:
            self._model = model
        else:
            print(f'loading wake model from default path {FileParams.default_model_path}')
            self._model = get_model_wrapper(FileParams.default_model_path)

    def check_wake(self, data: bytes) -> bool:
        """Checks if the data is the wake word.
        Args:
            data: the audio data to check
        Returns:
            True if the data is the wake word, False otherwise
        """
        mfccs = self._get_features(data)
        prediction: float = self._model.predict(mfccs)
        print(f'prediction: {round(prediction, 5):>10}', end='\r')
        triggered = self.trigger.check_trigger(prediction)
        if triggered:
            self._last_activation_audio = self.feature_audio
        return triggered

    def _get_features(self, buffer: bytes) -> np.ndarray:
        """Converts the audio buffer into MFCC features.
        Args:
            buffer: the audio buffer
        Returns:
            the MFCC features
        """
        # Convert the buffer to a normalized numpy array
        np_audio = np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0
        self.audio_buffer = np.concatenate((self.audio_buffer, np_audio))

        # If there are enough samples, extract new MFCC features
        if len(self.audio_buffer) >= self._ap.window_samples:
            # Convert audio into MFCC features
            new_features = self._get_mfccs(self.audio_buffer)

            # The number of audio frames used to create the new_features
            audio_frames_used: int = len(new_features) * self._ap.hop_samples

            # Save the audio associated with the new features (to save audio files of activations)
            new_features_audio: np.ndarray = self.audio_buffer[:audio_frames_used]
            self.feature_audio = np.concatenate((self.feature_audio[audio_frames_used:], new_features_audio))

            # Remove the samples that were used to create the new_features
            self.audio_buffer = self.audio_buffer[audio_frames_used:]

            # Make sure the new_features will fit in the model
            if len(new_features) > len(self.input_features):
                new_features = new_features[-len(self.input_features):]

            # Add the new features to the end of the existing features
            self.input_features = np.concatenate((self.input_features[len(new_features):], new_features))

        return self.input_features

    def _get_mfccs(self, window_audio: np.ndarray):
        return mfcc_spec(audio=window_audio,
                         sample_rate=self._ap.sample_rate,
                         window_stride=(self._ap.window_samples, self._ap.hop_samples),
                         num_filt=self._ap.n_filt,
                         fft_size=self._ap.n_fft,
                         num_coeffs=self._ap.n_mfcc)

    def last_activation_audio(self) -> bytes:
        """Returns the audio corresponding to the last wake word activation.
        Returns:
            the audio corresponding to the last wake word activation
        Raises:
            Exception: if there is no activation to return
        """
        if self._last_activation_audio is None:
            raise Exception('No activation to return')
        bytes_audio = (self._last_activation_audio *
                       32768.0).astype(np.int16).tobytes()
        return bytes_audio
