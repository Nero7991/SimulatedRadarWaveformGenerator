import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

def fft_vs_time(iq_samples, sample_rate=10e6):
    """
    Plot the FFT magnitude vs time for an array of IQ samples.

    Parameters:
        iq_samples (numpy.ndarray): 1D numpy array of IQ samples with shape (N,) where each element is a tuple of (real, imag) values.
        sample_rate (float): Sampling rate in Hz. Default is 10 MHz.

    Returns:
        None
    """
    # Convert the array of tuples to an array of complex numbers
    complex_samples = np.array([s[0] + 1j * s[1] for s in iq_samples])

    # Create the complex-valued numpy array
    iq_array = np.array(complex_samples, dtype=np.complex128)

    #Set FFT size
    fft_size = 1024

    # Compute the number of time slices and FFT size
    num_slices = int(len(iq_array) / fft_size)
    
    # Compute the FFT of each time slice
    spectra = np.zeros((num_slices, fft_size))
    for i in range(num_slices):
        start_idx = i * fft_size
        end_idx = start_idx + fft_size
        spectrum = np.abs(np.fft.fft(iq_array[start_idx:end_idx]))
        spectra[i] = spectrum

    # Create the waterfall plot
    fig, ax = plt.subplots(figsize=(10, 5))
    extent = (0, (num_slices - 1) * fft_size / sample_rate, 0, sample_rate / 2e6)
    ax.imshow(spectra.T, aspect='auto', extent=extent, cmap='jet')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('FFT vs Time Waterfall Plot')
    plt.show()

# Load the .mat file
mat = h5py.File('data/SimulatedRadarWaveforms/Group3/group3_subset_1.mat', 'r')

print(mat.keys())
#sys.exit()
n = int(input("Enter the index of sample\n"))


# Access the data
data = mat['group1_waveformSubset_1']

labels = mat['group1_radarStatusSubset_1']
print("Label = ")
print(labels[n])

# Convert the data to a numpy array
data = data[()]

# Print the data
print(data.shape)
print(data[:2])
print((data[n]).shape)
print((data[n])[0])
fft_vs_time(data[n])

# Close the file
mat.close()

