from keras.models import Sequential
from keras.layers import GRU, BatchNormalization, Dense
from keras.losses import binary_crossentropy

n_features = 29  # TODO
feature_size = 13  # TODO


def get_model():
    model = Sequential()
    model.add(GRU(
        units=20,
        activation='linear',
        input_shape=(n_features, feature_size),
        dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model
