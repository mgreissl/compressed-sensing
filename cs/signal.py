import numpy as np
from scipy.fft import dct, idct

def generate_sparse_signal(duration, sample_rate, frequency_parts):
    """
    This function generates a sparse signal by adding together multiple cosine functions with different frequencies.

    Parameters:
    - duration: float, total time duration of the signal in seconds.
    - sample_rate: int, the number of samples per second.
    - frequency_parts: list of int, list of frequencies in Hz that will be components of the signal.

    Returns:
    - sparse_function: ndarray, generated sparse signal.
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    sparse_function = np.sum([np.cos(2 * np.pi * freq * t) for freq in frequency_parts], axis=0)
    return sparse_function

def mse(original_signal, reconstructed_signal):
    """
    This function calculates the Mean Squared Error (MSE) between the original signal and the reconstructed signal.

    Parameters:
    - original_signal: ndarray, the original signal.
    - reconstructed_signal: ndarray, the reconstructed signal.

    Returns:
    - error: float, the mean squared error between the original and reconstructed signals.
    """
    return np.mean((original_signal - reconstructed_signal) ** 2)

def perform_dct(signal):
    """
    This function performs the Discrete Cosine Transform (DCT) on the input signal.

    Parameters:
    - signal: ndarray, the input signal.

    Returns:
    - transformed_signal: ndarray, the DCT transformed signal.
    """
    return dct(signal)

def perform_idct(frequency_representation):
    """
    This function performs the Inverse Discrete Cosine Transform (IDCT) on the frequency representation of a signal.

    Parameters:
    - frequency_representation: ndarray, the frequency representation of the signal.

    Returns:
    - signal: ndarray, the reconstructed signal in the time domain.
    """
    return np.real(idct(frequency_representation))