import wake
import os


FALSE_ACTIVATION_DIR = './wake/data/false_activations'
TRUE_ACTIVATION_DIR = './wake/data/true_activations'


class ActivationSaver:
    def __init__(self, sample_size:  int, sample_rate: int):
        self.pos_save_dir = TRUE_ACTIVATION_DIR
        self.neg_save_dir = FALSE_ACTIVATION_DIR

        # Create the save directories
        for dir in [self.pos_save_dir, self.neg_save_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)
        
        self.pos_save_idx = wake.audio_collection.get_greatest_index(
            self.pos_save_dir) + 1
        self.neg_save_idx = wake.audio_collection.get_greatest_index(
            self.neg_save_dir) + 1
        self.sample_size = sample_size
        self.sample_rate = sample_rate

    def save(self, audio: bytes, correct_activation: bool) -> None:
        """
        Saves the given audio as a true or false activation.
        Args:
            audio: the audio to save
            correct_activation: whether the activation was a correct activation
        """
        if correct_activation:
            save_dir = self.pos_save_dir
            index = self.pos_save_idx
            self.pos_save_idx += 1
        else:
            save_dir = self.neg_save_dir
            index = self.neg_save_idx
            self.neg_save_idx += 1

        wake.audio_collection.utils.save_wav_file(
            f'{save_dir}/activation-{index:04d}.wav',
            self.sample_size,
            self.sample_rate,
            audio)
        print(
            f'Saved {correct_activation} activation {index} to {save_dir}')
