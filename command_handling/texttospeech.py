from gtts import gTTS
import tempfile
from pygame import mixer
import os

DEFAULT_LANG = 'en'
DEFAULT_ACCENT = 'us'  # co.uk


class TextToSpeech:

    def __init__(self, on_pi: bool, lang=DEFAULT_LANG, tld=DEFAULT_ACCENT):
        self.lang = lang
        self.tld = tld
        self.speak = self.speak_pi if on_pi else self.speak_windows
        mixer.init()

    def speak(txt: str) -> None:
        """Speaks the given text."""
        pass

    def speak_windows(self, txt):
        try:
            speech = gTTS(text=txt, lang=self.lang, tld=self.tld)
            file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            filepath = file.name
            file.close()
            speech.save(filepath)
            mixer.music.load(filepath)
            mixer.music.play()
            print('Speaking...')
            while mixer.music.get_busy() == True:
                continue
            print('done')
        except Exception as e:
            print(e)
        finally:
            mixer.music.unload()
            os.remove(filepath)
                

    def speak_pi(self, txt):
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3')
            speech = gTTS(text=txt, lang=self.lang, tld=self.tld)
            speech.write_to_fp(temp_file)
            mixer.music.load(temp_file.name)
            mixer.music.play()
            print('before busy')
            while mixer.music.get_busy() == True:
                continue
            print('done')
        finally:
            if temp_file is not None:
                temp_file.close()

    def __del__(self):
        mixer.quit()


if __name__ == "__main__":
    import os
    txt = 'hello there. this is an example'
    on_pi = os.name != 'nt'
    print('On raspberry pi: ', on_pi)
    tts = TextToSpeech(on_pi)
    tts.speak(txt)
