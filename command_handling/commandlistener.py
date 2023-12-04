import audioop
from pygame import mixer
from wake.parameters import mycroftParams as ap

VOL_THRESH = 200  # minimum volume to start listening

END_QUIET_T = 1.0  # allowed silence while continuing to listen in seconds
END_QUIET_FRAMES = int(END_QUIET_T * ap.sample_rate / ap.chunk_size)

BEG_QUIET_T = 5  # allowed silence while waiting for command in seconds
BEG_QUIET_FRAMES = int(BEG_QUIET_T * ap.sample_rate / ap.chunk_size)

MAX_COMMAND_T = 10  # how long the command can be in seconds
MAX_COMMAND_FRAMES = int(MAX_COMMAND_T * ap.sample_rate / ap.chunk_size)

# Path to the sound to play when listening
WAKE_AUDIO_PATH = "./notification.mp3"


class CommandListener:
    """Class that listens for a command after the wake word is detected."""

    def __init__(self):
        mixer.init()
        self.beg_waiting_frames = 0
        self.quiet_frames = 0
        self.audio_buffer = []
        self.listening = False
        print(f"ALLOWED_QUIET_FRAMES = {END_QUIET_FRAMES}")
        print(f"MAX_COMMAND_FRAMES = {MAX_COMMAND_FRAMES}")

    def listen(self, data: bytes) -> bool:
        """
        Records the incoming audio and returns True when the command is complete.
        Args:
            data: the audio data to record
        Returns:
            True when the command is complete, False if still listening
        """
        volume = audioop.max(data, 2)

        # Handle before heard anything
        if not self.listening:
            if volume > VOL_THRESH:
                # Start listening if volume is above threshold
                self.listening = True
            elif self.beg_waiting_frames == 0:
                # Play sound the first time this is called
                print('Waiting to hear a command...')
                self._play_listening_sound()
            elif self.beg_waiting_frames > BEG_QUIET_FRAMES:
                self.beg_waiting_frames = 0
                return True  # Done listening
            self.beg_waiting_frames += 1

        if self.listening:
            self.audio_buffer.append(data)
            print(f"volume: {volume:<20}",
                  f"quiet frames: {self.quiet_frames:>5}/{END_QUIET_FRAMES}", end='\r')
            if volume < VOL_THRESH:
                self.quiet_frames += 1
            else:
                self.quiet_frames = 0

            if self.quiet_frames > END_QUIET_FRAMES:
                self.quiet_frames = 0
                self.listening = False
                if len(self.audio_buffer) > 0:
                    return True

            if len(self.audio_buffer) > MAX_COMMAND_FRAMES:
                self.listening = False
                return True

        return False

    def _play_listening_sound(self):
        """Plays a sound to indicate that the command listener is listening."""
        mixer.music.load(WAKE_AUDIO_PATH)
        mixer.music.play()

    def get_audio(self) -> bytes:
        """
        Gets the audio recorded by the command listener.
        Returns:
            the audio recorded by the command listener, or None if no audio was recorded
        """
        if len(self.audio_buffer) == 0:
            print('No audio to return')
            return None
        audio = b''.join(self.audio_buffer)
        self.audio_buffer = []
        self.quiet_frames = 0
        self.beg_waiting_frames = 0
        return audio
