from model import get_model
from data import get_data
from keras.callbacks import TensorBoard
from datetime import datetime

model = get_model()
model.summary()
input, output = get_data()

model.fit(
    x=input,
    y=output,
    batch_size=5000,
    epochs=1,
    callbacks=[TensorBoard("logdir")]
)

model.save(f'models/{datetime.now().strftime("%Y-%m-%d-%H.%M.%S")}-model.keras')