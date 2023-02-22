import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_signal(iq_array):
    # Calculate the power of the IQ samples
    power = np.abs(iq_array)**2

    # Calculate the noise floor as the median of the power
    noise_floor = np.mean(power)

    # Calculate the signal threshold as 2 standard deviations above the noise floor
    signal_threshold = noise_floor + 10 * np.sqrt(np.var(power))

    # Find the indices of the samples with power above the signal threshold
    signal_indices = np.where(power > signal_threshold)[0]

    # Find the start and end indices of the signal
    start_index = signal_indices[0]
    end_index = signal_indices[-1]

    # Find a few samples before and after the signal
    before_samples = min(start_index, 4096)
    after_samples = min(len(iq_array) - end_index - 1, 4096)

    # Adjust the start and end indices to ensure that the extracted signal has a length of at least 8192 samples
    while end_index - start_index + 1 + before_samples + after_samples < 8192:
        if start_index > 0:
            start_index -= 1
            before_samples += 1
        elif end_index < len(iq_array) - 1:
            end_index += 1
            after_samples += 1
        else:
            break

    # Extract the signal from the IQ samples
    signal = iq_array[start_index - before_samples:end_index + after_samples + 1]

    # Print some information about the extracted signal
    print('Original length:', len(iq_array))
    print('Signal length:', len(signal))
    print('Noise floor:', noise_floor)

    return signal

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

    #iq_array = extract_signal(iq_array)

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
mat = h5py.File('data/SimulatedRadarWaveforms/Group4/group4_subset_10.mat', 'r')

print(mat.keys())

keys = list(mat.keys())
#sys.exit()
while(True):
    n = int(input("Enter the index of sample\n"))
    labels = np.array(mat[keys[3]])
    labels = labels.reshape(-1)
    print("Label = ")
    print(labels[n])
    if(not labels[n]):
        if(input("Not a signal, continue (y/n): ") == "y"):
            break
    else:
        break

# Access the data
data = mat[keys[4]]

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

