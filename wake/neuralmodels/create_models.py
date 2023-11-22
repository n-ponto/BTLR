"""
Creates different model architectures for testing.
"""

from keras.models import Sequential
from keras.layers import GRU, LayerNormalization, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
# import tensorflow as tf

n_features = 29  # TODO
feature_size = 13  # TODO


def simple_model():
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

    optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                     beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model


def layer_norm_model():
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

    optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                     beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model


def variable_GRU_units_model(gru_units=20):
    """
    Test different GRU units
    """
    model = Sequential(name='model2')
    model.add(GRU(
        units=gru_units,
        activation='linear',
        input_shape=(n_features, feature_size),
        dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                     beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])

    return model


def variable_intermediate_layer_model(intermediate_units=5):
    """
    Another linear layer
    """
    model = Sequential()
    model.add(GRU(
        units=16,
        activation='linear',
        input_shape=(n_features, feature_size),
        dropout=0.2))

    # Intermediate layer 20 -> 5
    model.add(Dense(intermediate_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                     beta_2=0.999, epsilon=1e-7)
    model.compile(
        optimizer=optimizer,
        loss=binary_crossentropy,
        metrics=['accuracy'])
    return model


def cnn_model():
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    # norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    input_shape = (n_features, feature_size)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        # layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ], name='cnn_model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=binary_crossentropy,
        metrics=['accuracy'],
    )

    return model
