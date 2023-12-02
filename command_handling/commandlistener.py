import audioop
from wake.parameters import mycroftParams as ap

RMS_THRESHOLD = 200
ALLOWED_QUIET_T = 1.0  # allowed silence while continuing to listen in seconds
ALLOWED_QUIET_FRAMES = int(ALLOWED_QUIET_T * ap.sample_rate / ap.chunk_size)

MAX_COMMAND_T = 10  # how long the command can be in seconds
MAX_COMMAND_FRAMES = int(MAX_COMMAND_T * ap.sample_rate / ap.chunk_size)

class CommandListener:

    def __init__(self):
        self.quiet_frames = 0
        self.audio_buffer = []
        print(f"ALLOWED_QUIET_FRAMES = {ALLOWED_QUIET_FRAMES}")
        print(f"MAX_COMMAND_FRAMES = {MAX_COMMAND_FRAMES}")

    def listen(self, data: bytes) -> bool:
        self.audio_buffer.append(data)
        rms = audioop.rms(data, 2)
        print(f"rms: {rms:<20}", f"quiet frames: {self.quiet_frames:>5}/{ALLOWED_QUIET_FRAMES}", end='\r')
        if rms < RMS_THRESHOLD:
            self.quiet_frames += 1
        else:
            self.quiet_frames = 0

        if self.quiet_frames > ALLOWED_QUIET_FRAMES:
            self.quiet_frames = 0
            if len(self.audio_buffer) > 0:
                # self.audio_buffer = []
                return True
        
        if len(self.audio_buffer) > MAX_COMMAND_FRAMES:
            # self.audio_buffer = []
            return True
        
        return False

    def get_audio(self) -> bytes:
        audio = b''.join(self.audio_buffer)
        self.audio_buffer = []
        return audio