import threading
import time
import audio_collection, listener
from parameters import DEFAULT_AUDIO_PARAMS as AP
from modelwrapper import get_model_wrapper

# MODEL_PATH = './checkpoints/smaller_cnn'
MODEL_PATH = './trained_model.tflite'

model = get_model_wrapper(MODEL_PATH)
p, stream = audio_collection.utils.create_stream(audio_params=AP)
wake_listener = listener.WakeListener(model=model, ap=AP)

continue_listening = True

def thread_function():
    """Continuously passes audio data to the wake word listener."""
    print('Thread started...')
    while continue_listening:
        data = stream.read(AP.chunk_size, False)
        triggered = wake_listener.check_wake(data)
        if triggered:
            print('WAKE!!!' + " " * 100)

# Start the wake word listener in the background
thread = threading.Thread(target=thread_function, daemon=True)

try:
    print('\n\nPress enter to quit.\n\n')
    time.sleep(1)
    thread.start()
    text = input()  # Wait for user to press enter
    continue_listening = False  # Stop the thread
    print('Stopping...')
    thread.join()
except Exception as e:
    print(e)

print('closing stream...')
stream.close()
p.terminate()
