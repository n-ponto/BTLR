"""
Converts a model from Keras into TensorFlow Lite format.
"""
import tensorflow as tf
from tensorflow import keras


def convert_model(model_path: str, optimizations: list = None, save_path: str = None):
    # Load the model
    keras_model = keras.models.load_model(model_path)
    print('Loaded model')
    keras_model.summary()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        # Below needs to be disabled to run on raspberry pi
        # tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    if optimizations is not None and len(optimizations) > 0:
        converter.optimizations = optimizations

    tflite_model = converter.convert()

    # Save the model
    if save_path is None:
        # Get file name from path
        import os
        assert (os.path.isdir(model_path))
        save_path = os.path.basename(model_path) + '.tflite'
    if not save_path.endswith('.tflite'):
        save_path += '.tflite'
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Saved model to {save_path}')


# If this file is being run directly, convert the model
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to the keras model')
    args = parser.parse_args()
    convert_model(args.model_path, [tf.lite.Optimize.DEFAULT], 'trained_model.tflite')
