from tensorflow import keras
from listener.listener import Listener

MODEL_PATH = './trained_model'


if __name__ == '__main__':
    model = keras.models.load_model(MODEL_PATH)
    model.summary()
    listener = Listener(model)
    listener.start_listening()
