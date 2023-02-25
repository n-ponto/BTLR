'''
Logic for waking from sleep

https://github.com/Picovoice/porcupine#python
'''

import pvporcupine
from pvrecorder import PvRecorder
from voice import Voice
from speechText import SpeechText, EXIT_PHRASES


def main():
    with open('key.txt', 'r') as f:
        access_key = f.read()
    speech2text = SpeechText()
    print('Listening for wake word...')
    voice = Voice()

    try:
        wake_word_handle = pvporcupine.create(
        access_key=access_key, keywords=['porcupine'])
        recorder = PvRecorder(
        device_index=0, frame_length=wake_word_handle.frame_length)
        recorder.start()

        while True:
            audio = recorder.read()
            keyword_index = wake_word_handle.process(audio)
            if keyword_index >= 0:
                print('Detected')
                recorder.stop()
                text = speech2text.speechToText(show=True)
                if text is not None and text.lower() in EXIT_PHRASES:
                    exit()
                else:
                    voice.sayBlocking(text)

                recorder.start()
    except KeyboardInterrupt:
        pass
    finally:
        print('Closing listener')
        recorder.delete()
        wake_word_handle.delete()


if __name__ == '__main__':
    main()

'''
Refrences for building NN from scratch
https://www.tensorflow.org/tutorials/audio/simple_audioc
https://www.tensorflow.org/lite/microcontrollers
'''