import pyaudio
import wave
import os
from wake.collect.collect import AudioParams

print('Collecting noise...')
print('Press ctrl-c to end and save noise.')

p = pyaudio.PyAudio()

save_path = 'data/random'
save_index = 0

stream = p.open(
    rate=AudioParams.fs,
    channels=AudioParams.channels,
    format=AudioParams.format,
    frames_per_buffer=AudioParams.chunk_size,
    input=True
)
frames = []

try:
    while True:
        data = stream.read(AudioParams.chunk_size)
        frames.append(data)
except KeyboardInterrupt:
    print('\nsaving noise...')

joined_frames = b''.join(frames)

filename = f'noise-{save_index:02d}.wav'
filename = os.path.join(save_path, filename)
wf = wave.open(filename, 'wb')
wf.setnchannels(AudioParams.channels)
wf.setsampwidth(p.get_sample_size(AudioParams.format))
wf.setframerate(AudioParams.fs)
wf.writeframes(joined_frames)
wf.close()

print('Done!')