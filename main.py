import numpy as np
from cs.plotting import plot_original_signal, plot_reconstructed_signal, plot_error_values
from cs.optimization import reconstruct_signal
from cs.signal import generate_sparse_signal, mse, perform_dct
from cs.matrix import generate_measurement_matrix, perform_compressed_sensing_measurement

def main():
    np.random.seed(420)  # set seed for reproducibility

    n = 1000
    m_percentage_values = [5, 10, 15]  # different percentages of measurements

    error_values = []  # list to store the error values

    # generate sparse signal
    t = np.linspace(0.1, 0.2, n)
    frequency_parts = [30, 200]
    x = generate_sparse_signal(0.1, n / 0.1, frequency_parts)

    method = 'bernoulli' # 'bernoulli', 'gaussian' or 'equidistant'

    x_freq = perform_dct(x)

    for m_percentage in m_percentage_values:
        m = int(m_percentage / 100 * n)

        phi, perm = generate_measurement_matrix(n, m, matrix_type=method)
        z, theta = perform_compressed_sensing_measurement(phi, x_freq)

        x_rec = reconstruct_signal(theta, z)

        error = mse(x, x_rec)
        error_values.append(error)

        plot_original_signal(t, x, perm, savefig=f'figures/original_signal_{m_percentage}.pdf')
        plot_reconstructed_signal(t, x_rec, savefig=f'figures/reconstructed_signal_{m_percentage}.pdf')

    plot_error_values(m_percentage_values, error_values, savefig=f'figures/error_{method}.pdf')


if __name__ == "__main__":
    main()