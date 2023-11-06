from keras.models import Sequential
from keras.layers import GRU, LayerNormalization, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
# import tensorflow as tf

n_features = 29  # TODO
feature_size = 13  # TODO


def get_model1():
    """
    Base model
    """
    model = Sequential(name='model1')
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

def get_model2():
    """
    Uses Adam optimizer
    """
    model = Sequential(name='model2')
    model.add(GRU(
        units=20,
        activation='linear',
        input_shape=(n_features, feature_size),
        dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model

def get_model3():
    """
    Includes LayerNormalization
    """
    model = Sequential(name='model3')

    model.add(LayerNormalization(input_shape=(n_features, feature_size)))

    model.add(GRU(
        units=20,
        activation='linear',
        dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model