import wake
from enum import Enum
import commandlistener
import os
import speech_recognition as sr

ap = wake.parameters.DEFAULT_AUDIO_PARAMS

stop_keywords: list = ['stop', 'quit', 'exit', 'terminate', 'end', 'finish', 'done']

# Create the listener
p, stream = wake.create_stream(ap)
wake_listener = wake.WakeListener()
command_listener = commandlistener.CommandListener()

class ListeningState(Enum):
    ASLEEP = 0
    AWAKE = 1

current_state = ListeningState.ASLEEP
SAVE_DIR = './commands'
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"Created directory {SAVE_DIR}")

save_index = wake.utils.get_greatest_index(SAVE_DIR) + 1

recognizer = sr.Recognizer()

continue_listening = True

while continue_listening:
    data = stream.read(ap.chunk_size, False)
    if current_state == ListeningState.ASLEEP:
        triggered = wake_listener.check_wake(data)
        if triggered:
            print('WAKE!!!')
            current_state = ListeningState.AWAKE
    else:
        command_done = command_listener.listen(data)
        if command_done:
            print('Command done!')
            command_audio = command_listener.get_audio()
            # For now save the audio to a file
            # # filename = f"{SAVE_DIR}\\activation-{save_index}.wav"
            # # wake.utils.save_wav_file(filename, p.get_sample_size(ap.format), ap.sample_rate, command_audio)
            # # print(f"Saved activation {filename}")
            # save_index += 1
            print('Waiting on speech recognition...')
            audio_data = sr.AudioData(command_audio, ap.sample_rate, p.get_sample_size(ap.format))
            text = recognizer.recognize_google(audio_data).lower()
            print(text)
            current_state = ListeningState.ASLEEP
            if any([keyword in text for keyword in stop_keywords]):
                continue_listening = False
                print('Stopping...')