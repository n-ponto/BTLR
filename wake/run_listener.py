from listener.listener import Listener
from neuralmodels.modelwrapper import get_model_wrapper, ModelWrapper
import parameters

# MODEL_PATH = './checkpoints/smaller_cnn'
MODEL_PATH = './trained_model.tflite'

if __name__ == '__main__':
    model: ModelWrapper = get_model_wrapper(MODEL_PATH)
    audio_params = parameters.mycroftParams
    listener = Listener(model, audio_params)
    listener.start_listening()
