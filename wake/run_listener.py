from listener.listener import Listener
from model.modelwrapper import get_model_wrapper, ModelWrapper

# MODEL_PATH = './checkpoints/model_GRU_20'
MODEL_PATH = './trained_model.tflite'

if __name__ == '__main__':
    model: ModelWrapper = get_model_wrapper(MODEL_PATH)
    listener = Listener(model)
    listener.start_listening()
