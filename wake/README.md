# Wake

All the logic for the wake word listening engine. This is a lightweight neural network keyword detection engine to run locally on a Raspberry Pi.

## Videos

### Theory and Overview

https://www.youtube.com/watch?v=QmUSeGU-9vQ&t=432s


### In-Depth Walkthrough of the Code

https://www.youtube.com/watch?v=5EkY-kvej1g

## Using the pretrained model

The `trained_model.tflite` model is already trained for the wake word *"Hey Plato"*, as in the Greek philosopher Plato. I haven't tested this model with other people's voices, so I'm not sure how well it will work for you.

This code should work out of the box using
```
run_listener.py
```

# Training your own model

**Summary of steps for training a new model:**

1. Use `collect_samples.py` to collect postive, negative, and noise samples
2. (OPTIONAL) Download more noise data, I used the [Mozilla Common Voice](https://commonvoice.mozilla.org/en) dataset. Make sure to put a seperate folder in `./data/downloaded_data_folder` with the downloaded .WAV or .MP3 files.
3. Use `check_samples.py` to make sure the positive samples are correct
4. Use `split_data.py` to create .CSV files for the train, validation, and test datasets, respectively.
5. Use `preprocess_dataset.py` to convert the audio files into .NPY files with features and labels for the model
6. Use `train.ipynb` to create and train the neural network
7. (OPTIONAL) convert the model to TensorFlow Lite using `convert_model.py`
8. Run the listener with `run_listener.py`

## Collect Samples

```
python collect_samples.py --help
```

Use this to collect **positive** samples of saying the wake word, **negative** samples of other phrases similar to but different from the wake word, and **noise** samples of anything else (that's not the wake word).

These samples will show up in seperate folders in `./data`, for example `./data/neg`

```
python check_samples.py
```

Before continuting, use `check_samples.py` to ensure that all the samples in `./data/pos` are good representations of the wake word (no empty files)

### (OPTIONAL) download additional noises

Download more noise data, I used the [Mozilla Common Voice](https://commonvoice.mozilla.org/en) dataset. I made the assumption there were no recordings of someone saying "Hey Plato".

Make sure to put a seperate folder in `./data/downloaded_data_folder` with the downloaded .WAV or .MP3 files.

## Preprocess Data

```
python split_data.py
```

This will create three seperate CSV files in `./data`, one each for train, validation, and test.

```
python preprocess_dataset.py val
```

This will create `.data/val_x.npy` and `.data/val_y.npy`. Call the script again for train and test sets. Start with the validation set because it's much smaller than train, and if there are any issues it will be easier to diagnose. 

Those .NPY files are the features and labels for training the model. 

## Train the model

Walk through the steps of `train.ipynb` to load the features, create the model, and train it. There's also an example of using the model to make a prediction on one of the samples.

### OPTIONAL: convert the model with TF Lite

```
python convert_model.py ./checkpoints/simple_cnn
```

This will convert the model from Keras to TensorFlow Lite. Keras can only run on PC, while TensorFlow Lite models can run on either a PC or Pi. This will create `trained_model.tflite`

## Run the Listener

```
python run_listener.py
```

Runing the listener is as simple as this. 

**NOTE**: If you haven't optimized the model and don't have the `trained_model.tflite` file, you'll need to update the constant at the top of this file. There's already a comment showing how to use the unoptimized Keras version of the model. 