import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.fft import dct, idct
from cs.signal import generate_sparse_signal, mse
from cs.optimization import lasso
from cs.matrix import generate_measurement_matrix, perform_compressed_sensing_measurement
from cs.plotting import TUMRed, TUMOrange, TUMGreen

np.random.seed(420)  # set seed for reproducibility

n = 1000
m_values = list(range(50, 850, 50))  # different numbers of measurements

error_values_1 = []  # list to store the error values for alpha = 1
error_values_5 = []  # list to store the error values for alpha = 5
error_values_10 = []  # list to store the error values for alpha = 10

# generate sparse signal
t = np.linspace(0.1, 0.2, n)
frequency_parts = [30, 200]
x = generate_sparse_signal(0.1, n / 0.1, frequency_parts)

# discrete cosine transform
x_freq = dct(x)

for m in m_values:
    phi, _ = generate_measurement_matrix(n, m, matrix_type='bernoulli')
    z, theta = perform_compressed_sensing_measurement(phi, x_freq)

    x_rec_1 = np.real(idct(lasso(theta, z, alpha=1)))
    x_rec_5 = np.real(idct(lasso(theta, z, alpha=5)))
    x_rec_10 = np.real(idct(lasso(theta, z, alpha=10)))

    error_1 = mse(x, x_rec_1)
    error_values_1.append(error_1)

    error_5 = mse(x, x_rec_5)
    error_values_5.append(error_5)

    error_10 = mse(x, x_rec_10)
    error_values_10.append(error_10)

m_percentage = np.array(m_values) / n * 100

# create an error plot
plt.figure(figsize=(10, 6))
plt.plot(m_percentage, error_values_1, color=TUMRed, marker='o', label=r'$\alpha=1$', linewidth=1.5)
plt.plot(m_percentage, error_values_5, color=TUMOrange, marker='s', label=r'$\alpha=5$', linewidth=1.5)
plt.plot(m_percentage, error_values_10, color=TUMGreen, marker='^', label=r'$\alpha=10$', linewidth=1.5)
plt.xlabel('Measurements in % of the sample size')
plt.ylabel('Logarithmic error')
# set y-axis to log scale
plt.yscale('log')
# format x-axis labels as percentages
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))
plt.legend()
plt.tight_layout()
plt.savefig('figures/error_comp_alpha.pdf')

# show the plot
plt.show()

# save the data
np.save('data/error_comp_alpha.npy', np.array([m_percentage, error_values_1, error_values_5, error_values_10]))

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
             + r"    & \multicolumn{3}{c}{Penalty Parameter} \\" + "\n" \
             + r"    Subsample & $\alpha=1$ & $\alpha=5$ & $\alpha=10$ \\" + "\n" \
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

export_to_tex('tables/error_comp_alpha.tex', 'data/error_comp_alpha.npy')