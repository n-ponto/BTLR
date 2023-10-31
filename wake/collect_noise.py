import pyaudio
import wave
import os
from parameters import AudioParams

print('Collecting noise...')
print('Press ctrl-c to end and save noise.')

p = pyaudio.PyAudio()

FILE_LENGTH_SECONDS = 30

save_path = 'data\\random'

if not os.path.isdir(save_path):
    os.mkdir(save_path)


def get_file_index():
    files = os.listdir(save_path)
    if len(files) < 1:
        return 0
    files.sort()

    last_index = int(files[-1][:-4].split('-')[1])
    print(f'starting from index {last_index}')
    return last_index + 1


def save(save_index: int, joined_frames: bytes):
    filename = f'noise-{save_index:02d}.wav'
    filename = os.path.join(save_path, filename)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(AudioParams.channels)
    wf.setsampwidth(p.get_sample_size(AudioParams.format))
    wf.setframerate(AudioParams.sample_rate)
    wf.writeframes(joined_frames)
    wf.close()
    print(f'Saved {filename}')


def loop(initial_index: int):
    chunks_per_file = AudioParams.sample_rate//AudioParams.chunk_size*FILE_LENGTH_SECONDS
    index = initial_index
    frames = []
    try:
        while True:
            for _ in range(chunks_per_file):
                data = stream.read(AudioParams.chunk_size)
                frames.append(data)
            joined_frames = b''.join(frames)
            save(index, joined_frames)
            frames = []
            index += 1
    except KeyboardInterrupt:
        print('interrupted...')
    if len(frames) > 0:
        joined_frames = b''.join(frames)
        save(index+1, joined_frames)


stream = p.open(
    rate=AudioParams.sample_rate,
    channels=AudioParams.channels,
    format=AudioParams.format,
    frames_per_buffer=AudioParams.chunk_size,
    input=True
)
initial_index = get_file_index()
loop(initial_index)


print('Done!')
