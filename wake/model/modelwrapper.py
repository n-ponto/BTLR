"""
Makes it easier to use different models for prediction without changing the code.
"""
import numpy as np


def is_pi():
    """
    Check if running on a raspberry pi.
    Returns True if running on a raspberry pi, False otherwise.
    """
    import os
    os_name = os.name
    print(f'os_name = {os_name}')
    return os.name != 'nt'


class ModelWrapper:
    """
    Wrapper around the model to make it easier to use different models for 
    prediction without changing the code.
    """

    def predict(self, input) -> float:
        """
        Predicts if the input corresponds to the wake word.
        Args:
            input: the input to the model
        Returns:
            A float between 0 and 1, where 0 means the input is not the wake word, and 1 means it is.
        """
        raise NotImplementedError


def get_model_wrapper(model_path: str) -> ModelWrapper:
    """
    Returns a model wrapper object based on the model path.
    """
    running_on_pi = is_pi()
    if model_path.endswith('.tflite'):
        return TFLiteModelWrapper(model_path, running_on_pi)
    else:
        assert (
            not running_on_pi), f'Cannot use Keras model on raspberry pi: {model_path}.\nPlease convert the model to a tensorflow lite model.'
        return KerasModelWrapper(model_path)


class KerasModelWrapper(ModelWrapper):
    """
    Predictor for a keras model
    """

    def __init__(self, model_path):
        import keras
        self._model: keras.Model = keras.models.load_model(model_path)
        print('Using Keras model')
        self._model.summary()

    def predict(self, input_data) -> float:
        return self._model.predict(input_data[np.newaxis], verbose=0)[0][0]


class TFLiteModelWrapper(ModelWrapper):
    """
    Predictor for a tensorflow lite model
    """

    def __init__(self, model_path: str, on_pi: bool):
        # Load the tensorflow package depending on the platform
        tflite = TFLiteModelWrapper._load_tensorflow_package(on_pi)
        self._interpreter = tflite.Interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._input_index = self._interpreter.get_input_details()[0]['index']
        self._output_index = self._interpreter.get_output_details()[0]['index']

    def predict(self, input) -> float:
        # Plug the input into the interpreter
        self._interpreter.set_tensor(
            self._input_index, input.astype(np.float32)[np.newaxis])

        # Predict using the interpreter
        self._interpreter.invoke()

        # Get and return the result
        output_data = self._interpreter.get_tensor(self._output_index)
        return output_data[0][0]

    @staticmethod
    def _load_tensorflow_package(on_pi: bool):
        """
        Loads different tensorlfow packages depending on the platform.
        Loads tensorflow light runtime for raspberry pi, and tensorflow for desktop.
        """
        if on_pi:
            import tflite_runtime.interpreter as tflite
        else:
            import tensorflow as tf
            tflite = tf.lite
        return tflite
