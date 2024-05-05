"""
Function for augmenting spectrograms to generate additional synthetic data

From:
https://www.kaggle.com/code/davids1992/specaugment-quick-implementation
"""

import numpy as np

def spec_augment(mfcc: np.ndarray, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """Applies spectrogram augmentation to the MFCC
    Args:
        mfcc: MFCC data
        num_mask: number of masks to apply
        freq_masking_max_percentage: maximum percentage of frequencies to mask
        time_masking_max_percentage: maximum percentage of time to mask
    Returns:
        MFCC data with spectrogram augmentation applied
    """
    assert (len(mfcc.shape) == 2), f'Expected 2D array, got {mfcc.shape}'
    mfcc = mfcc.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = mfcc.shape
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        mfcc[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(
            low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        mfcc[t0:t0 + num_frames_to_mask, :] = 0

    return mfcc