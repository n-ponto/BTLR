from gtts import gTTS
from tempfile import NamedTemporaryFile
import pygame
import utils

DEFAULT_LANG = 'en'
DEFAULT_ACCENT = 'co.uk'


class TextToSpeech:

    def __init__(self, on_pi: bool, lang=DEFAULT_LANG, tld=DEFAULT_ACCENT):
        self.lang = lang
        self.tld = tld
        self.speak = self.speak_pi if on_pi else self.speak_windows
        if on_pi:
            pygame.mixer.init()

    def speak(txt: str) -> None:
        pass

    def speak_windows(self, txt):
        print(f'[SPEAKING]: {txt}')

    def speak_pi(self, txt):
        try:
            temp_file = NamedTemporaryFile()
            speech = gTTS(text=txt, lang=self.lang, tld=self.tld)
            speech.write_to_fp(temp_file)
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            print('before busy')
            while pygame.mixer.music.get_busy() == True:
                continue
            print('done')
        finally:
            if temp_file is not None:
                temp_file.close()
            pass

    def __del__(self):
        pygame.mixer.quit()


if __name__ == "__main__":
    txt = 'hello there. this is an example'
    on_pi = utils.is_pi()
    print('On raspberry pi: ', on_pi)
    tts = TextToSpeech(on_pi)
    tts.speak(txt)
