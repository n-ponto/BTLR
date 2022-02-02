'''
Text to speech
Contains all logic for converting text to speech for the user
'''
import pyttsx3 as tts
from speechText import SpeechText


class Voice:

    _engine: tts.Engine

    def __init__(self):
        self._engine = tts.init('sapi5')  # For windows, might need to change
        voices = self._engine.getProperty('voices')
        # 0 for male, 1 for female
        self._engine.setProperty('voice', voices[1].id)
        self._engine.setProperty('rate', 190)  # Speech words per minute
        self._engine.say("Initializing voice")
        self._engine.runAndWait()

    def __del__(self):
        self._engine.stop()

    def printSettings(self):
        print(f"Speech rate {self._engine.getProperty('rate')}")
        print(f"Volume {self._engine.getProperty('volume')}")

    def sayBlocking(self, text: str):
        self._engine.say(text)
        self._engine.runAndWait()


def echo():
    print("Echoing user phrases")
    from speechText import SpeechText, exit_phrases
    print("Say any of the following to quit")
    print(exit_phrases)
    st = SpeechText()
    voice = Voice()
    while True:
        text = st.speechToText(show=True)
        if text is None:
            continue
        if any(phrase in text.lower() for phrase in exit_phrases):
            break
        voice.sayBlocking(text)
    print("\nDone echoing")


if __name__ == "__main__":
    import sys
    if (len(sys.argv) > 1):
        if sys.argv[1].lower() == 'echo':
            echo()


'''
Refrences
https://pypi.org/project/pyttsx3/

Documentation
https://pyttsx3.readthedocs.io/en/latest/engine.html?highlight=init#pyttsx3.init
'''
