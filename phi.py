import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.fft import dct
from cs.optimization import reconstruct_signal
from cs.signal import generate_sparse_signal, mse
from cs.matrix import generate_measurement_matrix, perform_compressed_sensing_measurement
from cs.plotting import TUMRed, TUMOrange, TUMGreen

np.random.seed(420)  # set seed for reproducibility

n = 1000
m_values = list(range(50, 850, 50))  # different numbers of measurements

error_values_bernoulli = []  # list to store the error values for Bernoulli matrix
error_values_gaussian = []  # list to store the error values for Gaussian matrix
error_values_equidistant = []  # list to store the error values for equidistant matrix

# generate sparse signal
t = np.linspace(0.1, 0.2, n)
frequency_parts = [30, 200]
x = generate_sparse_signal(0.1, n / 0.1, frequency_parts)

# discrete cosine transform
x_freq = dct(x)

for m in m_values:
    phi_bernoulli, _ = generate_measurement_matrix(n, m, matrix_type='bernoulli')
    phi_gaussian, _ = generate_measurement_matrix(n, m, matrix_type='gaussian')
    phi_equidistant, _ = generate_measurement_matrix(n, m, matrix_type='equidistant')

    z_bernoulli, theta_bernoulli = perform_compressed_sensing_measurement(phi_bernoulli, x_freq)
    z_gaussian, theta_gaussian = perform_compressed_sensing_measurement(phi_gaussian, x_freq)
    z_equidistant, theta_equidistant = perform_compressed_sensing_measurement(phi_equidistant, x_freq)

    x_rec_bernoulli = reconstruct_signal(theta_bernoulli, z_bernoulli)
    x_rec_gaussian = reconstruct_signal(theta_gaussian, z_gaussian)
    x_rec_equidistant = reconstruct_signal(theta_equidistant, z_equidistant)

    error_bernoulli = mse(x, x_rec_bernoulli)
    error_values_bernoulli.append(error_bernoulli)

    error_gaussian = mse(x, x_rec_gaussian)
    error_values_gaussian.append(error_gaussian)

    error_equidistant = mse(x, x_rec_equidistant)
    error_values_equidistant.append(error_equidistant)

m_percentage = np.array(m_values) / n * 100

# create an error plot
plt.figure(figsize=(10, 6))
plt.plot(m_percentage, error_values_bernoulli, color=TUMRed, marker='o', label='MSE (Bernoulli)', linewidth=1.5)
plt.plot(m_percentage, error_values_gaussian, color=TUMOrange, marker='s', label='MSE (Gaussian)', linewidth=1.5)
plt.plot(m_percentage, error_values_equidistant, color=TUMGreen, marker='^', label='MSE (Deterministic)', linewidth=1.5)
plt.xlabel('Measurements in % of the sample size')
plt.ylabel('Logarithmic error')
# set y-axis to log scale
plt.yscale('log')
# format x-axis labels as percentages
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
plt.legend()
plt.tight_layout()
plt.savefig('figures/error_comp_phi.pdf')

# show the plot
plt.show()

# save the data
np.save('data/error_comp_phi.npy', np.array([m_percentage, error_values_bernoulli, error_values_gaussian, error_values_equidistant]))

# export data to latex table
def export_to_tex(filename, data):
    """
    Exports numerical data to a LaTeX tabular format and writes it to a tex-file.

    Parameters:
    - filename: str, the name of the file where the file will be written.
    - data: str, the path to a .npy file containing an array of data to be converted.

    The output LaTeX table consists of four columns with headers. The data is expected to match these headers.
    """
    header = r"% automatically generated from " + data + "\n" \
             + r"\begin{tabular}{l|l|l|l}" + "\n" \
             + r"    & \multicolumn{3}{c}{Sensing Matrix} \\" + "\n" \
             + r"    Subsample & Bernoulli & Gaussian & Deterministic \\" + "\n" \
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

export_to_tex('tables/error_comp_phi.tex', 'data/error_comp_phi.npy')