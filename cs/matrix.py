import numpy as np
from scipy.fft import dct

def generate_measurement_matrix(n, m, matrix_type='bernoulli'):
    """
    This function generates a measurement matrix based on the specified type.

    Parameters:
    - n: int, the number of columns in the matrix.
    - m: int, the number of rows in the matrix.
    - matrix_type: str (optional), the type of matrix to generate. Options are 'bernoulli', 'gaussian', and 'equidistant'.
      Default is 'bernoulli'.

    Returns:
    - phi: ndarray, the generated measurement matrix.
    - perm: ndarray or None, permutation vector used for the 'bernoulli' matrix. None for the other types.
    """
    if matrix_type == 'bernoulli':
        phi = np.identity(n)
        perm = np.random.permutation(n)[:m]
        phi = phi[perm, :]
        return phi, perm

    elif matrix_type == 'gaussian':
        phi = np.random.randn(m, n) / np.sqrt(m)
        return phi, None

    elif matrix_type == 'equidistant':
        phi = np.zeros((m, n))
        step_size = n // m
        for i in range(m):
            phi[i, i * step_size] = 1
        return phi, None

    else:
        raise ValueError("Invalid matrix_type. Expected 'bernoulli', 'gaussian' or 'equidistant'.")

def perform_compressed_sensing_measurement(Phi, x_freq, noise=False):
    """
    This function performs compressed sensing measurement on a given signal.

    Parameters:
    - Phi: ndarray, the measurement matrix.
    - x_freq: ndarray, the frequency domain representation of the signal to measure.
    - noise: bool (optional), if True, adds Gaussian noise to the measurements. Default is False.

    Returns:
    - z: ndarray, the compressed sensing measurements.
    - theta: ndarray, the combined measurement and transform matrix.
    """
    psi = dct(np.identity(Phi.shape[1]), axis=0)  # Calculate the Discrete Cosine Transform matrix.
    theta = Phi @ psi  # Compute the combined measurement and transform matrix.
    z = theta @ x_freq  # Perform the compressed sensing measurement.

    if noise:
        m = Phi.shape[0]
        z += np.random.randn(m) / np.sqrt(m)  # noise with standard deviation 1/sqrt(m)

    return z, theta
