'''
Script to remove bad samples from the positive sample directory.

Plays each file and asks if it should be deleted.
'''

from pydub import AudioSegment
from pydub.playback import play
import os

dir = "C:\\Users\\noah\\repos\\BTLR\\wake\\data\\pos"

for file in os.listdir(dir):
    path = os.path.join(dir, file)
    audio = AudioSegment.from_wav(path)
    play(audio)
    if input(file + ": ") != "":
        print('deleting')
        os.remove(path)
