import pyaudio
import wave
import audioop
import os
from parameters import FileParams, AudioParams

# How loud the audio needs to be to trigger activation
THRESHOLD_MULTIPLIER = 1.5  # threshold louder than ambient noise
THRESHOLD_SECONDS = 2  # time to calculate threshold
THRESHOLD = 400

ALLOWED_SILENCE_SECONDS = 0.3  # allowed time of silence to remain activated

class SampleCollector:

    def __init__(self, args):
        self.negative = args.negative
        self.save_directory = self.get_directory(args.dir)
        self.save_index = self.get_file_index(args.overwrite)
        self.save_file_prefix = FileParams.neg_file_name if args.negative else FileParams.pos_file_name
        self.sample_size = None

    def get_directory(self, sampledir: str):
        if not os.path.isdir(sampledir):
            os.mkdir(sampledir)
        
        subdir = FileParams.neg_sample_dir if self.negative else FileParams.pos_sample_dir
        fulldir = os.path.join(sampledir, subdir)

        if not os.path.isdir(fulldir):
            os.mkdir(fulldir)
        
        return fulldir
    
    def get_file_index(self, overwrite):
        if overwrite: return 0

        files = os.listdir(self.save_directory)
        if len(files) < 1: return 0
        files.sort()

        last_index = int(files[-1][:-4].split('-')[1])
        print(f'starting from index {last_index}')
        return last_index + 1

    def collect_wakeword(self, stream, rms_threshold):
        if self.negative:
            print("Please say words OTHER than the wake word...")
        else:
            print("Please say the wake word...")

        allowed_silence_frames = int(AudioParams.sample_rate/AudioParams.chunk_size*ALLOWED_SILENCE_SECONDS)
        silence_frames = 0
        collecting = False
        frames = []
        i = 0
        while True:
            data = stream.read(AudioParams.chunk_size)
            rms = audioop.rms(data, 2)
            print(str(rms) + '   \t', end='')
            if not collecting and rms > rms_threshold:
                # New sound activates collection
                print('activated\t', end='')
                collecting = True
            elif collecting and rms <= rms_threshold:
                # Silence during collection
                if silence_frames > allowed_silence_frames:
                    # Exceeded allowed time, save the sample
                    print('saving   \t', end='')
                    self.save_data(b''.join(frames))
                    i += 1
                    collecting = False
                    frames = []
                    silence_frames = 0
                else:
                    # Record that it was silent
                    silence_frames += 1
            elif collecting and rms > rms_threshold:
                # If there was a sound, reset silence frames
                silence_frames = 0
            
            print('', end='\r')
            if collecting:
                frames.append(data)
            else:
                # Always keep one prefix frame
                frames = [data]


    def save_data(self, data):
        filename = f'{self.save_file_prefix}-{self.save_index:02d}.wav'
        filename = os.path.join(self.save_directory, filename)
        self.save_index += 1
        wf = wave.open(filename, 'wb')
        wf.setnchannels(AudioParams.channels)
        wf.setsampwidth(self.sample_size)
        wf.setframerate(AudioParams.sample_rate)
        wf.writeframes(data)
        wf.close()

    def run(self):
        p = pyaudio.PyAudio()
        self.sample_size = p.get_sample_size(AudioParams.format)

        stream = p.open(
            rate=AudioParams.sample_rate,
            channels=AudioParams.channels,
            format=AudioParams.format,
            frames_per_buffer=AudioParams.chunk_size,
            input=True
        )
        # rms_threshold = self.get_rms_threshold(stream)
        rms_threshold = THRESHOLD

        try:
            self.collect_wakeword(stream, rms_threshold)
        except KeyboardInterrupt:
            print('\n\nending process...')

        stream.stop_stream()
        stream.close()

    
    @staticmethod
    def get_rms_threshold(stream) -> int:
        print("Please stay quiet...")
        thresh_steps = int(AudioParams.sample_rate/AudioParams.chunk_size*THRESHOLD_SECONDS)
        thresh_max = None
        for _ in range(thresh_steps):
            data = stream.read(AudioParams.chunk_size)
            rms = audioop.rms(data, 2)
            if not thresh_max or thresh_max < rms:
                thresh_max = rms
        rms_threshold = thresh_max * THRESHOLD_MULTIPLIER
        print(f'threshold is {rms_threshold}')
        return rms_threshold


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', default=FileParams.default_data_dir, nargs='?')
    parser.add_argument('-n', '--negative', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    sample_collector = SampleCollector(args)
    sample_collector.run()
