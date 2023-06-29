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

def perform_compressed_sensing_measurement(phi, x_freq, noise=False):
    """
    This function performs compressed sensing measurement on a given signal.

    Parameters:
    - phi: ndarray, the measurement matrix.
    - x_freq: ndarray, the frequency domain representation of the signal to measure.
    - noise: bool (optional), if True, adds Gaussian noise to the measurements. Default is False.

    Returns:
    - z: ndarray, the compressed sensing measurements.
    - theta: ndarray, the combined measurement and transform matrix.
    """
    psi = dct(np.identity(phi.shape[1]), axis=0)  # Calculate the Discrete Cosine Transform matrix.
    theta = phi @ psi  # Compute the combined measurement and transform matrix.
    z = theta @ x_freq  # Perform the compressed sensing measurement.

    if noise:
        m = phi.shape[0]
        z += np.random.randn(m) / np.sqrt(m)  # noise with standard deviation 1/sqrt(m)

    return z, theta

def coherence(matrix):
    """
    This function calculates the coherence of a given matrix.

    Parameters:
    - matrix: ndarray, the input matrix.

    Returns:
    - mu: float, the coherence of the input matrix.
    """
    n = matrix.shape[1]
    mu = 0

    # Normalize the columns of the matrix
    matrix_normalized = matrix / np.linalg.norm(matrix, axis=0)

    for i in range(n):
        for j in range(i+1, n):
            # Calculate the absolute inner product of columns i and j
            inner_product = np.abs(np.inner(matrix_normalized[:, i], matrix_normalized[:, j]))

            # Update maximum inner product if necessary
            if inner_product > mu:
                mu = inner_product

    return mu
