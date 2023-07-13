import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np

# define TUM corporate colors
TUMBlue = (0/255, 101/255, 189/255)
TUMRed = (196/255, 7/255, 27/255)
TUMGreen = (162/255, 173/255, 0/255)
TUMOrange = (227/255, 114/255, 34/255)
TUMWhite = (255 / 255, 255 / 255, 255 / 255)
TUMBlack = (0 / 255, 0 / 255, 0 / 255)

def plot_original_signal(t, signal, measurements_indices=None, savefig=None):
    """
    Plots the original signal along with the measurement points (optional).

    Parameters:
    - t: ndarray, the time points.
    - signal: ndarray, the signal values at the time points.
    - measurements_indices: list (optional), indices of the measurement points.
    - savefig: str (optional), the file path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, color=TUMBlue, label='Original Signal', linewidth=1.5)
    if measurements_indices is not None:
        plt.scatter(t[measurements_indices], signal[measurements_indices], color=TUMRed, marker='x', label='Measurements')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()

def plot_reconstructed_signal(t, signal, savefig=None):
    """
    Plots the reconstructed signal.

    Parameters:
    - t: ndarray, the time points.
    - signal: ndarray, the signal values at the time points.
    - savefig: str (optional), the file path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, color=TUMBlue, label='Reconstructed Signal', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()

def plot_error_values(m_percentage_values, error_values, savefig=None):
    """
    Plots error values for different measurement percentages.

    Parameters:
    - m_percentage_values: list, the measurement percentages.
    - error_values: list, the corresponding error values.
    - savefig: str (optional), the file path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(m_percentage_values, error_values, color=TUMRed, marker='o', label='MSE', linewidth=1.5)
    plt.xlabel('Measurements in % of the Sample Size')
    plt.ylabel('Logarithmic Error')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
    plt.legend(loc='upper right')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()

def plot_error_heatmap(m_percentage_values, sparsity_values, error_values, savefig=None):
    """
    Plots a heatmap of error values for different sparsity and measurement percentages.

    Parameters:
    - m_percentage_values: list, the measurement percentages.
    - sparsity_values: list, the sparsity values.
    - error_values: ndarray, the corresponding error values.
    - savefig: str (optional), the file path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(error_values, cmap=colors.LinearSegmentedColormap.from_list(TUMWhite, [TUMBlack, TUMBlue, TUMWhite]), origin='lower', aspect='auto', norm=colors.LogNorm(vmin=0.000001, vmax=1))
    plt.xticks(np.arange(len(sparsity_values)), sparsity_values)
    plt.xlabel('Sparsity')
    plt.yticks(np.arange(len(m_percentage_values)), [f'{val}%' for val in m_percentage_values])
    plt.ylabel('Measurements')
    cbar = plt.colorbar()
    cbar.set_label('Logarithmic Error')
    cbar.set_ticks([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()
