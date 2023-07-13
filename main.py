import numpy as np
from cs.plotting import plot_original_signal, plot_reconstructed_signal, plot_error_values
from cs.reconstruct import reconstruct_signal
from cs.signal import generate_sparse_signal, mse, perform_dct
from cs.matrix import generate_measurement_matrix, perform_compressed_sensing_measurement

def main():
    np.random.seed(420)  # set seed for reproducibility

    n = 1000
    m_percentage_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # different percentages of measurements

    error_values = []  # list to store the error values

    # generate sparse signal
    t = np.linspace(0.1, 0.2, n)
    frequency_parts = [30, 200]
    x = generate_sparse_signal(0.1, n / 0.1, frequency_parts)

    sensing = 'bernoulli' # 'bernoulli', 'gaussian' or 'equidistant'
    method = 'omp' # 'basis_pursuit', 'lasso' or 'omp'

    x_freq = perform_dct(x)

    for m_percentage in m_percentage_values:
        m = int(m_percentage / 100 * n)

        phi, mask = generate_measurement_matrix(n, m, matrix_type=sensing)
        z, theta = perform_compressed_sensing_measurement(phi, x_freq)

        x_rec = reconstruct_signal(theta, z, method=method)

        error = mse(x, x_rec)
        error_values.append(error)

        plot_original_signal(t, x, mask, savefig=f'figures/original_signal_{m_percentage}.pdf')
        plot_reconstructed_signal(t, x_rec, savefig=f'figures/reconstructed_signal_{m_percentage}.pdf')

    plot_error_values(m_percentage_values, error_values, savefig=f'figures/error_{method}.pdf')

    # save the data
    np.save(f'data/error_{method}.npy', np.array([m_percentage_values, error_values]))

    def export_to_tex(filename, data):
        """
        Exports numerical data to a LaTeX tabular format and writes it to a tex-file.

        Parameters:
        - filename: str, the name of the file where the file will be written.
        - data: str, the path to a .npy file containing an array of data to be converted.

        The output LaTeX table consists of four columns with headers. The data is expected to match these headers.
        """
        header = r"% automatically generated from " + data + "\n" \
                 + r"\begin{tabular}{l|l}" + "\n" \
                 + r"    Subsample & \\" + "\n" \
                 + r"    \hline"

        footer = r"\end{tabular}"

        data_arr = np.load(data)

        with open(filename, 'w') as file:
            file.write(header + "\n")

            for i in range(len(data_arr[0])):
                row = f"    {data_arr[0][i]}\\% & "
                row += " & ".join([f"${x:.4e}$" for x in data_arr[1:, i]])
                row += r" \\"
                file.write(row + "\n")

            file.write(footer)

    export_to_tex(f'tables/error_{method}.tex', f'data/error_{method}.npy')


if __name__ == "__main__":
    main()