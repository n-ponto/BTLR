import wake
import speech_recognition as sr
from command_handling import CommandListener, CommandHandler
import utils

# Create the listener
ap = wake.parameters.DEFAULT_AUDIO_PARAMS
pyaudio, stream = wake.audio_collection.utils.create_stream(ap)
wake_listener = wake.WakeListener()
command_listener = CommandListener()
sample_size = pyaudio.get_sample_size(ap.format)
command_handler = CommandHandler(
    wake_listener, sample_size, ap.sample_rate, utils.is_pi(), True)
state_asleep = True
recognizer = sr.Recognizer()


def speech_to_text(recognizer: sr.Recognizer, audio: bytes) -> str:
    """
    Converts speech to text.
    Args:
        recognizer: the speech recognizer
        audio: the audio data to convert
    Returns:
        the text
    """
    if audio is None:
        print('No command audio')
        return ""
    audio_data = sr.AudioData(audio, ap.sample_rate,
                              pyaudio.get_sample_size(ap.format))
    try:
        text = recognizer.recognize_google(audio_data).lower()
    except sr.UnknownValueError:
        print('Could not understand audio')
        return
    except sr.RequestError as e:
        print(f'Request error: {e}')
        return
    return text


while True:
    data = stream.read(ap.chunk_size, False)
    if state_asleep:
        triggered = wake_listener.check_wake(data)
        if triggered:
            print('WAKE!!!')
            state_asleep = False  # Wake up and listen for command
    else:
        command_done = command_listener.listen(data)
        if command_done:
            print('Command done!')
            state_asleep = True  # Go back to sleep after processing command
            command_audio = command_listener.get_audio()

            # Convert speech to text
            command_text = speech_to_text(recognizer, command_audio)
            print(command_text)
            command_handler.handle(command_text)
