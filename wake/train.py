import model
import data
from keras.callbacks import TensorBoard
from datetime import datetime
from noise import get_noise

my_model = model.get_model()
my_model.summary()
input, output = data.get_data()
noise_input, noise_output = get_noise()

import numpy as np
input = np.concatenate([input, noise_input])
output = np.concatenate([output, noise_output])
assert(len(input) == len(output))
print(f'samples {len(input)}')

my_model.fit(
    x=input,
    y=output,
    batch_size=5000,
    epochs=60,
    callbacks=[TensorBoard("logdir")]
)

my_model.save(f'models/{datetime.now().strftime("%Y-%m-%d-%H.%M.%S")}-model.keras')