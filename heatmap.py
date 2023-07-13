import numpy as np
from cs.plotting import plot_error_heatmap
from cs.reconstruct import reconstruct_signal
from cs.signal import generate_sparse_signal, mse, perform_dct
from cs.matrix import generate_measurement_matrix, perform_compressed_sensing_measurement

np.random.seed(420)  # set seed for reproducibility

n = 1000  # dimension of y
m_percentage_values = [5, 10, 15, 20, 25, 30, 35, 40]  # different percentages of measurements
sparsity_values = [2, 5, 10, 15, 20, 50, 100]  # different sparsity values

error_values = np.zeros((len(m_percentage_values), len(sparsity_values)))  # list to store the error values

for i, m_percentage in enumerate(m_percentage_values):
    m = int(m_percentage / 100 * n)

    for j, sparsity in enumerate(sparsity_values):
        frequency_parts = np.random.choice(range(1, 200), sparsity, replace=False)
        x = generate_sparse_signal(0.1, n / 0.1, frequency_parts)

        x_freq = perform_dct(x)

        phi, _ = generate_measurement_matrix(n, m)
        z, theta = perform_compressed_sensing_measurement(phi, x_freq)

        x_rec = reconstruct_signal(theta, z)

        error = mse(x, x_rec)
        error_values[i, j] = error

plot_error_heatmap(m_percentage_values, sparsity_values, error_values, savefig='figures/error_heatmap.pdf')

# save the data
np.save('data/error_heatmap.npy', error_values)
