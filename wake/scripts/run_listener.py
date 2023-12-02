import threading
import time
from context import audio_collection, listener, neuralmodels, parameters

# MODEL_PATH = './checkpoints/smaller_cnn'
MODEL_PATH = '../trained_model.tflite'

if __name__ == '__main__':
    model = neuralmodels.get_model_wrapper(MODEL_PATH)
    audio_params: parameters.AudioParams = parameters.mycroftParams
    p, stream = audio_collection.utils.create_stream(ap=audio_params)
    wake_listener = listener.WakeListener(model=model, ap=audio_params)

    continue_listening = True

    def thread_function():
        """Continuously passes audio data to the wake word listener."""
        print('Thread started...')
        while continue_listening:
            data = stream.read(audio_params.chunk_size, False)
            triggered = wake_listener.check_wake(data)
            if triggered:
                print('WAKE!!!')

    # Start the wake word listener in the background
    thread = threading.Thread(target=thread_function, daemon=True)

    try:
        print('Press enter to quit')
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
