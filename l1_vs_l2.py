import numpy as np
import matplotlib.pyplot as plt
from cs.reconstruct import basis_pursuit
from cs.plotting import TUMBlue

np.random.seed(420)  # set seed for reproducibility

# solve z = theta * y
n = 1000  # dimension of y
m = 100  # number of measurements
Theta = np.random.randn(m, n)
z = np.random.randn(m)

# l1 minimum norm solution y_l1
y_l1 = basis_pursuit(Theta, z, norm=1)

# l2 minimum norm solution y_l2
y_l2 = basis_pursuit(Theta, z, norm=2)

# create the plots and histograms
plots = ['l1', 'l2']

for plot_name in plots:
    plt.figure()
    plt.plot(locals()[f"y_{plot_name}"], color=TUMBlue, linewidth=1.5)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(f'figures/{plot_name}.pdf')

    plt.figure()
    plt.hist(locals()[f"y_{plot_name}"], bins=np.arange(-0.105, 0.105, 0.01), rwidth=0.95, color=TUMBlue)
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.ylim(0, 700)
    plt.savefig(f'figures/{plot_name}_hist.pdf')

    plt.figure()
    plt.boxplot(locals()[f"y_{plot_name}"], patch_artist=True,
                medianprops={'color': 'black'},
                boxprops={'facecolor': TUMBlue, 'color': TUMBlue})
    plt.xticks([])
    plt.ylabel('Amplitude')
    plt.ylim(-0.05, 0.05)
    plt.yticks([-0.05, -0.01, 0, 0.01, 0.05])
    plt.savefig(f'figures/{plot_name}_box.pdf')

# show the plots
plt.show()