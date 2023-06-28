import numpy as np
from scipy.optimize import minimize
from scipy.fft import idct

def basis_pursuit(theta, z, y0=None, norm=1):
    """
    This function implements the basis pursuit algorithm.

    Parameters:
    - Theta: ndarray, the measurement matrix.
    - z: ndarray, the measured samples.
    - y0: ndarray (optional), initial guess for the solution. Default is a random vector.
    - norm: int (optional), the order of the norm to minimize. Default is 1 (l1 norm).

    Returns:
    - ndarray, the frequency representation of the reconstructed signal.
    """
    if not (norm >= 0 or norm == np.inf):
        raise ValueError("norm must be greater than or equal to 0 or inf.")
    constr = ({'type': 'eq', 'fun': lambda y: theta @ y - z})
    if y0 is None:
        y0 = np.random.randn(theta.shape[1])  # initialize y
    res = minimize(lambda x: np.linalg.norm(x, ord=norm), y0, method='SLSQP', constraints=constr)
    return res.x

def reconstruct_signal(theta, z):
    """
    This function reconstructs a signal from its measured samples using the basis pursuit algorithm,
    and transforms it back to the time domain using the inverse discrete cosine transform (IDCT).

    Parameters:
    - Theta: ndarray, the measurement matrix.
    - z: ndarray, the measured samples.

    Returns:
    - ndarray, the reconstructed signal in the time domain.
    """
    reconstructed_freq = basis_pursuit(theta, z)
    return np.real(idct(reconstructed_freq))
