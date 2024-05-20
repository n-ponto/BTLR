import datetime
from .activationsaver import ActivationSaver
from .texttospeech import TextToSpeech
from .weather import Weather
from .blinds import Blinds


class Keywords:
    stop: set = {'stop', 'quit', 'exit', 'terminate', 'end', 'finish', 'done'}
    false_activation: set = {f'false {w}' for w in ['wake', 'activation', 'trigger']}
    weather: set = {'weather', 'temperature'}
    blinds: set = {'blinds', 'blind', 'window', 'windows'}


class CommandHandler:
    def __init__(self, wake_listener, sample_size: int, sample_rate: int, is_pi: bool, save_activations: bool):
        """
        Creates a new CommandHandler.
        Args:
            wake_listener: the wake listener to use (`wake.WakeListener` instance)
            sample_size: the sample size (Bytes) of the audio
            sample_rate: the sample rate (Hz) of the audio
        """
        self.wake_listener = wake_listener
        self.save_activations = save_activations
        if self.save_activations:
            self.activation_saver = ActivationSaver(sample_size=sample_size,
                                                    sample_rate=sample_rate)
        else:
            self.activation_saver = None
        self.tts = TextToSpeech(is_pi)
        self.weather = Weather()

    def handle(self, command_text: str) -> None:
        """Handles the given command text."""
        if self._keywords_in_command(command_text, Keywords.stop):
            print('Stopping...')
            exit(0)
        elif self._keywords_in_command(command_text, Keywords.false_activation):
            print('False activation detected')
            self._save_last_activation(False)
        elif self._keywords_in_command(command_text, Keywords.weather):
            weather = self.weather.get_weather()
            self.tts.speak(weather)
        elif self._keywords_in_command(command_text, Keywords.blinds):
            output: str = Blinds.handle(command_text)
            self.tts.speak(output)
        elif 'date' in command_text:
            today = datetime.date.today()
            readable_date = today.strftime("%B %d, %Y")
            self.tts.speak(readable_date)
        elif 'time' in command_text:
            now = datetime.datetime.now()
            readable_time = now.strftime("%I:%M %p")
            self.tts.speak(readable_time)
        else:
            print('Unknown command')
            return
        # If it understood the command, save as a true activation
        self._save_last_activation(True)

    def _save_last_activation(self, correct_activation: bool):
        """
        Saves the last activation from the wake listener as either a true or false activation.
        Args:
            correct_activation: whether the activation was a correct activation
        """
        if self.wake_listener is None or self.save_activations is False:
            return
        activation_audio = self.wake_listener.last_activation_audio()
        self.activation_saver.save(activation_audio, correct_activation)

    @staticmethod
    def _keywords_in_command(text: str, keywords: set) -> bool:
        """
        Checks whether the text contains any of the keywords.
        Args:
            text: the text to check
            keywords: the keywords to check for
        Returns:
            True if the text contains any of the keywords, False otherwise
        """
        return any([w in text for w in keywords])
