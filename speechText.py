'''
Logic for translating text to speech. When called it will use the microphone to
listen to speech and translate that speech into text
'''
import speech_recognition as sr
from time import time
from utils import *

exit_phrases = ["quit", "exit", "leave", "stop"]


def printUserSpeech(text: str):
    '''
    Formats and prints text to indicate it came from the user
    '''
    print(f'{ConsoleColors.OKCYAN}[USER] {text}{ConsoleColors.ENDC}')


class SpeechText:

    _r: sr.Recognizer
    _mic: sr.Microphone

    def __init__(self, mic=None):
        self._r = sr.Recognizer()
        if mic is not None:
            self.setMicrophone(mic)
        else:
            self._mic = sr.Microphone()
        self._r.pause_threshold = 0.8  # seconds of silence that indicate end of phrase
        # self._r.energy_threshold = 1000
        with self._mic as source:
            self._r.adjust_for_ambient_noise(source, duration=0.1)

    def listMicrophones(self):
        '''
        Prints a list of all the microphones 
        '''
        for i, micName in enumerate(sr.Microphone.list_microphone_names()):
            print(f"[{i}] \"{micName}\"")

    def _getMicIndex(self, searchName: str) -> int:
        '''
        Takes a string name, searches the list of mics for that name
        Returns index if found, None if no mic of that name is found
        '''
        for i, micName in enumerate(sr.Microphone.list_microphone_names()):
            if searchName.lower() == micName.lower():
                return i
        warning(f"Couldn't find mic of name: {searchName}")
        return None

    def setMicrophone(self, mic) -> bool:
        '''
        Takes either the name of a microphone or the index
        Returns true if the microphone was successfully set, false on failure
        '''
        if (type(mic) == int):
            self._mic = sr.Microphone(device_index=mic)
            return True
        elif(type(mic) == str):
            micIdx = self._getMicIndex(mic)
            if micIdx is not None:
                self._mic = sr.Microphone(device_index=micIdx)
                return True
        else:
            warning(f"Unexpected type for setting mic: {type(mic)}")
        return False

    def calibrateForAmbientNoise(self, duration=1):
        '''
        Recalibrates the energy threshold by taking a sample of the current noise
        '''
        prevThresh: float = self._r.energy_threshold
        with self._mic as source:
            self._r.adjust_for_ambient_noise(source, duration=duration)
        newThresh: float = self._r.energy_threshold
        print(f"Changed threshold from {prevThresh} to {newThresh}")

    def _recordAudio(self, duration=5) -> sr.AudioData:
        '''
        Records 'duration' seconds of audio from the microphone
        '''
        print("Recording...")
        with self._mic as source:
            return self._r.record(source, duration=duration)

    def _listenForAudioBlocking(self) -> sr.AudioData:
        '''
        Blocking call which will wait to hear sound
        '''
        print("Listening...")
        # self._r.energy_threshold = 1600
        with self._mic as source:
            try:
                start = time()
                audio: sr.AudioData = self._r.listen(source, phrase_time_limit=5)
                    # source,
                    # timeout=10, # Seconds to wait to hear a sound before timeout
                    # phrase_time_limit=8) # Seconds before phrase is cutoff
                print(f'duration {time()-start}')
                return audio
            except sr.WaitTimeoutError:
                print("Didn't hear anything")
                return None

    def _audioFromFile(self, filepath: str) -> sr.AudioData:
        '''
        Loads audio from a file path and loads it into text
        '''
        print(f"Loading audio file {filepath}...")
        with sr.AudioFile(filepath) as source:
            print(f"\tfound file with {source.DURATION} seconds of audio")
            audio = self._r.record(source)
        return audio

    def speechToText(self, show=False) -> str:
        '''
        Gets audio from the microphone and translates the audio into text
        Returns the interpreted text
        '''
        audio_data = self._listenForAudioBlocking()
        if audio_data is not None:
            print("Recognizing...")
            try:
                start = time()
                text = self._r.recognize_google(audio_data)
                print(f"duration {time()-start} sec")
                if show: printUserSpeech(text)
                return text
            except sr.UnknownValueError:
                error("Could not interpret audio")
        return None


if __name__ == "__main__":
    print("Speech recognition")
    st = SpeechText()
    while True:
        text = st.speechToText(show=True)
        if text is None:
            continue
        query = text.lower()
        if any(phrase in query for phrase in exit_phrases):
            break

    print("\nDone recognizing speech")


'''
Refrences
https://www.thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python
https://www.geeksforgeeks.org/voice-assistant-using-python/
https://realpython.com/python-speech-recognition/

SpeechRecognition library documentation
https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst

'''
