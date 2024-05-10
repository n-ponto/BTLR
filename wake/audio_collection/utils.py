import pyaudio

def save_wav_file(filename: str, sample_size: int, sample_rate: int, data: bytes) -> None:
    """Saves a wav file.
    Args:
        filename: the name of the file to save
        sample_size: the size of each sample in bytes
        data: the data to save
    """
    import wave
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_size)
    wf.setframerate(sample_rate)
    wf.writeframes(data)
    wf.close()


def create_stream(audio_params) -> (pyaudio.PyAudio, pyaudio.Stream):
    """Creates a pyaudio stream to record audio.
    Args:
        audio_params: `parameters.AudioParams` the audio parameters
    Returns:
        the pyaudio object and the stream object
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=audio_params.sample_rate,
        channels=1,
        format=audio_params.format,
        frames_per_buffer=audio_params.chunk_size,
        input=True
    )
    return p, stream


def get_greatest_index(dir: str) -> int:
    """Gets the largest file index in the directory.
    Args:
        dir: the directory to search
    Returns:
        the index of the last file in the directory
    """
    import os
    files = os.listdir(dir)
    if len(files) < 1:
        return -1
    files.sort()

    try:
        last_file = files[-1]
        name = last_file[:-4]
        number = name.split('-')[1]
        last_index = int(number)
        return last_index
    except Exception as e:
        print(f'error getting index from last file in directory {dir}')
        return -1
