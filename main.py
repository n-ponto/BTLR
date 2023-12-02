import wake
import speech_recognition as sr
from command_handling import CommandListener, CommandHandler

# Create the listener
ap = wake.parameters.DEFAULT_AUDIO_PARAMS
pyaudio, stream = wake.audio_collection.utils.create_stream(ap)
wake_listener = wake.WakeListener()
command_listener = CommandListener()
sample_size = pyaudio.get_sample_size(ap.format)
command_handler = CommandHandler(wake_listener, sample_size, ap.sample_rate)
state_asleep = True
recognizer = sr.Recognizer()

continue_listening = True

while continue_listening:
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
            try:
                print('Waiting on speech recognition...')
                audio_data = sr.AudioData(
                    command_audio, ap.sample_rate, pyaudio.get_sample_size(ap.format))
                text = recognizer.recognize_google(audio_data).lower()
            except sr.UnknownValueError:
                print('Could not understand audio')
                continue
            except sr.RequestError as e:
                print(f'Request error: {e}')
                continue

            print(text)
            command_handler.handle(text)
