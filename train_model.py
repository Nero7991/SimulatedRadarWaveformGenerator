import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import os
import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Define the number of groups and subsets
num_groups = 2
num_subsets = 2

SAMPLE_SIZE = 1000
POINTS_PER_SAMPLE = 100000
QUAD_SAMPLING_RATE = 10e6
FREQ_RES = QUAD_SAMPLING_RATE / POINTS_PER_SAMPLE

def display_complex_image(complex_array):
    """
    Displays a complex64 array as an image using the magnitude of the complex values as pixel intensities.
    
    Parameters:
    complex_array (np.ndarray): Input complex64 array to display as an image
    
    Returns:
    None
    """
    # Compute the magnitude of the complex values
    mag_array = np.abs(complex_array)
    
    # Normalize the pixel intensities between 0 and 1
    mag_array_norm = mag_array / np.max(mag_array)

    #mag_array_norm = mag_array_norm.reshape(100, -1)
    
    # Display the image
    plt.imshow(mag_array_norm, cmap='gray')
    plt.axis('off')
    plt.show()

def rolling_average_complex(arr, window_size):
    """
    Computes the rolling average on a np.complex64 array.

    Parameters:
    -----------
    arr : np.complex64 array
        Input array.
    window_size : int
        Size of the rolling window.

    Returns:
    --------
    out : np.complex64 array
        Output array with the same shape as input array.
    """
    # Split real and imaginary parts
    arr_real = np.real(arr)
    arr_imag = np.imag(arr)

    # Compute rolling average of real and imaginary parts separately
    arr_real_avg = np.convolve(arr_real, np.ones(window_size)/window_size, mode='same')
    arr_imag_avg = np.convolve(arr_imag, np.ones(window_size)/window_size, mode='same')

    # Combine the real and imaginary parts into a complex array
    out = arr_real_avg + 1j*arr_imag_avg

    return out

#iq_data = np.empty((1000, 100000))
label_data = np.empty((SAMPLE_SIZE, 1))

mat = h5py.File('RadarWaveformSet_2023-02-27_19-50.mat', 'r')

print(mat.keys())
keys = list(mat.keys())

print(mat[keys[3]])

labels = np.array(mat[keys[3]])
ref_data = np.array(mat[keys[4]])
labels = labels.reshape(-1)
labels_processed = np.empty([1000], dtype=int)
iq_data = np.empty([1000, 100000], dtype=np.float64)

for i in range(1000):
    labels_processed[i] = int(mat[labels[i]][0])
    waveform_data = np.array(mat[ref_data[0][i]])
    complex_samples = np.array([s[0] + 1j * s[1] for s in waveform_data[0]])
    complex_np_samples = np.array(complex_samples, dtype=np.complex64)
    iq_averaged = rolling_average_complex(complex_np_samples, 1)
    # iq_data[i] = rolling_average_complex(np.fft.fft(np.array(complex_samples, dtype=np.complex64)), 4)
    #Set FFT size
    fft_size = 500

    # Compute the number of time slices and FFT size
    num_slices = int(len(iq_averaged) / fft_size)
    
    # Compute the FFT of each time slice
    spectra = np.zeros((num_slices, fft_size))
    for j in range(num_slices):
        start_idx = j * fft_size
        end_idx = start_idx + fft_size
        spectrum = np.abs(np.fft.fft(iq_averaged[start_idx:end_idx]))
        spectra[j] = spectrum
    print(spectra.shape)
    #display_complex_image(spectra)
    iq_data[i] = spectra.flatten()
    print("Shape of array:", spectra.shape)
    print("Data type of array:", spectra.dtype)
    print("Size of array:", spectra.size)
    
    # freqs = np.arrange(POINTS_PER_SAMPLE) * FREQ_RES
    # print(freqs.shape)
    print("Converting data to complex numbers: " + str(i+1) + " of 1000")

print(labels_processed[:5])
print(iq_data[:5])

# # Load the .mat file
# mat = h5py.File('RadarWaveformSet_2023-02-26_19-59.mat', 'r')

# print(mat.keys())

# keys = list(mat.keys())

# labels = np.array(mat[keys[3]])
# labels = labels.reshape(-1)
# data = []
# data = mat[keys[4]][:]
# iq_data = np.empty((1000, 100000), dtype=np.complex128)

# for i in range(1000):
#     # Convert the array of tuples to an array of complex numbers
#     print("Converting data to complex numbers: " + str(i+1) + " of 200")
#     complex_samples = np.array([s[0] + 1j * s[1] for s in data[i]])

#     # Create the complex-valued numpy array
#     iq_array = np.array(complex_samples, dtype=np.complex128)

#     iq_data[i] = iq_array
# # Loop over each group
# for k in range(num_groups):
#     print("Group: " + str(k+1) + " of " + str(num_groups))
    
#     # Loop over each subset
#     for j in range(num_subsets):
#         # Define the file path
#         file_path = 'dataset/Group' + str(k+1) + '/group' + str(k+1) + '_subset_' + str(j+1) + '.npz'
        
#         # Check if the file exists
#         if os.path.exists(file_path):
#             # Load the data from the file
#             data = np.load(file_path)
#             if(k==0 and j==0):
#                 iq_data = np.fft.fft(data['X'])
#                 label_data = data['y']
#             else:
#                 # Compute the FFT of the IQ samples
#                 fft_data = np.fft.fft(iq_data)

#                 iq_data = np.concatenate((fft_data, data['X']), axis=0)

#                 label_data = np.concatenate((label_data, data['y']), axis=0)
            
#             # Print the shape of the data arrays
#             print("Subset " + str(j+1) + " of " + str(num_subsets) + ":")
#             print("  iq_data shape: " + str(iq_data.shape))
#             print("  label_data shape: " + str(label_data.shape))
#         else:
#             print("Subset " + str(j+1) + " of " + str(num_subsets) + " not found.")

# set random seed for reproducibility
np.random.seed(42)

# # load the IQ data and labels
# iq_data = np.load("iq_data.npy")
# label_data = np.load("label_data.npy")


sample_len = label_data.shape[0]
print("Sample length: " + str(sample_len))

# # Take the first 5% (500kHz BW) of the frequency bins
# num_freq_bins = len(fft_data)
# fft_data = fft_data[:num_freq_bins // 20]

# # convert complex IQ data to magnitude and phase
# iq_mag = np.abs(iq_data)
# iq_phase = np.angle(iq_data)

# # concatenate magnitude and phase data along the channel axis
# iq_data = np.concatenate((iq_mag, iq_phase), axis=1)

max_val = np.max(np.abs(iq_data))
iq_data = iq_data / max_val

# # flatten the data along the example and time axes
# iq_data = iq_data.reshape((sample_len, -1))
# label_data = label_data.reshape((sample_len, 1))

label_data = labels_processed

# iq_data = iq_data.flatten()
# print(iq_data[:2])

print(label_data.shape)

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(iq_data, label_data, test_size=0.2, random_state=42)

# Print the shapes of the resulting arrays
print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))

# build the RNN model
model = Sequential()
model.add(Dense(2048, input_shape=(100000,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# # compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # train the model
# model.fit(iq_data, label_data, epochs=10, batch_size=32, validation_split=0.1)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# create a TensorBoard callback with weight histograms
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
