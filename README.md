# Compressed Sensing Toolbox

This Python package provides a set of utilities for working with compressed sensing that allows for the reconstruction of sparse signals from fewer samples than typically required by the Nyquist-Shannon sampling theorem.

The package includes methods for generating sparse signals, creating different types of measurement matrices (Bernoulli, Gaussian, Equidistant), and reconstructing signals using basis pursuit. Additionally, it also provides a set of functions to visualize signals and errors with plotting functionalities.

This code was used to generate the figures and results in my thesis titled "Compressive Sensing - Theory and Applications".

## Getting Started

You can install this package by cloning the repository to your local machine:

```bash
git clone https://github.com/mgreissl/compressive-sensing.git
cd compressive-sensing
```

## Example

Here is a brief example of how you might use this package to create a sparse signal, perform compressed sensing measurements, and then reconstruct the signal:

```python
# generate a sparse signal
duration = 1.0  # seconds
sample_rate = 1000  # Hz
frequency_parts = [50, 120]  # Hz
signal = generate_sparse_signal(duration, sample_rate, frequency_parts)

# perform DCT
signal_freq = perform_dct(signal)

# generate a measurement matrix and perform measurements
m = 500  # number of measurements
n = len(signal)  # signal length
phi, measurements_indices = generate_measurement_matrix(n, m, matrix_type='bernoulli')

# perform compressed sensing measurement
z, theta = perform_compressed_sensing_measurement(phi, signal_freq)

# reconstruct the signal
signal_rec = reconstruct_signal(theta, z)

# calculate the Mean Square Error (MSE) between the original and reconstructed signals
error = mse(signal, signal_rec)
```

This package also includes functions for visualizing the original and reconstructed signals, as well as the errors:

```python
t = np.linspace(0, duration, len(signal))

# plot the original and reconstructed signals
plot_original_signal(t, signal, measurements_indices=measurements_indices)
plot_reconstructed_signal(t, signal_rec)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
