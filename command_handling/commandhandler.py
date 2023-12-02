import wake
from .activationsaver import ActivationSaver


class Keywords:
    stop: set = {'stop', 'quit', 'exit', 'terminate', 'end', 'finish', 'done'}
    false_activation: set = {f'false {w}' for w in [
        'wake', 'activation', 'trigger']}
    weather: set = {'weather', 'temperature'}

class CommandHandler:
    def __init__(self, wake_listener: wake.WakeListener, sample_size: int, sample_rate: int):
        """
        Creates a new CommandHandler.
        Args:
            wake_listener: the wake listener to use
            sample_size: the sample size (Bytes) of the audio
            sample_rate: the sample rate (Hz) of the audio
        """
        self.wake_listener = wake_listener
        self.activation_saver = ActivationSaver(sample_size=sample_size,
                                                sample_rate=sample_rate)

    def handle(self, command_text: str) -> None:
        """Handles the given command text."""
        if self._text_contains(command_text, Keywords.stop):
            print('Stopping...')
            exit(0)
        elif self._text_contains(command_text, Keywords.false_activation):
            print('False activation detected')
            self._save_last_activation(False)
        elif self._text_contains(command_text, Keywords.weather):
            print('Weather command detected')
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
        activation_audio = self.wake_listener.get_last_activation()
        self.activation_saver.save(activation_audio, correct_activation)

    @staticmethod
    def _text_contains(text: str, keywords: set) -> bool:
        """
        Checks whether the text contains any of the keywords.
        Args:
            text: the text to check
            keywords: the keywords to check for
        Returns:
            True if the text contains any of the keywords, False otherwise
        """
        return any([w in text for w in keywords])
