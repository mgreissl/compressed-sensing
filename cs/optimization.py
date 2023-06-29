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

def reconstruct_signal(theta, z, method='basis_pursuit'):
    """
    This function reconstructs a signal from its measured samples using the basis pursuit algorithm,
    and transforms it back to the time domain using the inverse discrete cosine transform (IDCT).

    Parameters:
    - Theta: ndarray, the measurement matrix.
    - z: ndarray, the measured samples.

    Returns:
    - ndarray, the reconstructed signal in the time domain.
    """
    if method == 'basis_pursuit':
        reconstructed_freq = basis_pursuit(theta, z)
    elif method == 'lasso':
        reconstructed_freq = lasso(theta, z)
    else:
        raise ValueError("method must be either 'basis_pursuit' or 'lasso'.")
    return np.real(idct(reconstructed_freq))

def lasso(theta, z, y0=None, alpha=1.0):
    """
    This function implements the LASSO algorithm.

    Parameters:
    - theta: ndarray, the measurement matrix.
    - z: ndarray, the measured samples.
    - y0: ndarray (optional), initial guess for the regression coefficients. Default is a random vector.
    - alpha: float (optional), the penalty parameter. Default is 1.0.

    Returns:
    - ndarray, the frequency representation of the reconstructed signal.
    """
    if y0 is None:
        y0 = np.random.randn(theta.shape[1])  # initialize y

    # the objective function to be minimized
    obj_func = lambda y: 1/2 * (np.linalg.norm(theta @ y - z, 2))**2 + alpha * np.linalg.norm(y, 1)

    # no constraints for LASSO
    res = minimize(obj_func, y0, method='SLSQP')
    return res.x