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

# Define the number of groups and subsets
num_groups = 2
num_subsets = 2

iq_data = np.empty((1000, 100000))
label_data = np.empty((1000, 1))

# Load the .mat file
mat = h5py.File('RadarWaveformSet_2023-02-26_19-59.mat', 'r')

print(mat.keys())

keys = list(mat.keys())

labels = np.array(mat[keys[3]])
labels = labels.reshape(-1)
data = []
data = mat[keys[4]][:]
iq_data = np.empty((1000, 100000), dtype=np.complex128)

for i in range(1000):
    # Convert the array of tuples to an array of complex numbers
    print("Converting data to complex numbers: " + str(i+1) + " of 200")
    complex_samples = np.array([s[0] + 1j * s[1] for s in data[i]])

    # Create the complex-valued numpy array
    iq_array = np.array(complex_samples, dtype=np.complex128)

    iq_data[i] = iq_array
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

# convert complex IQ data to magnitude and phase
iq_mag = np.abs(iq_data)
iq_phase = np.angle(iq_data)

# concatenate magnitude and phase data along the channel axis
iq_data = np.concatenate((iq_mag, iq_phase), axis=1)

# flatten the data along the example and time axes
iq_data = iq_data.reshape((sample_len, -1))
label_data = label_data.reshape((sample_len, 1))

iq_data = iq_data.reshape(-1, 800000, 2)

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(iq_data, label_data, test_size=0.4, random_state=42)

# Print the shapes of the resulting arrays
print("X_train shape: " + str(X_train.shape))
print("y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("y_test shape: " + str(y_test.shape))

# build the RNN model
model = Sequential()
model.add(LSTM(32, input_shape=(800000, 2)))
model.add(Dense(1, activation='sigmoid'))

# # compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # train the model
# model.fit(iq_data, label_data, epochs=10, batch_size=32, validation_split=0.1)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
