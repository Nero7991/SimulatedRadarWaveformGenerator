import numpy as np
import os
import h5py


# Define the number of groups and subsets
num_groups = 4
num_subsets = 20

for k in range(num_groups):
    print("Group: " + str(k+1) + " of 4")
    iq_data = []
    #label_data = []
    for j in range(num_subsets):
        # Load the data from the .mat file
        print("Opening file: " + str(j+1) + " of 50")
        file_path = 'data/SimulatedRadarWaveforms/Group' + str(k+1) + '/group' + str(k+1) + '_subset_' + str(j+1) + '.mat'
        print(file_path)

        # Load the data from the .mat file
        mat = h5py.File(file_path, 'r')

        # Access the keys
        keys = list(mat.keys())

        # Access the data using the keys
        labels = np.array(mat[keys[3]])
        labels = labels.reshape(-1)
        data = mat[keys[4]]
        iq_data = np.empty((200, 800000), dtype=np.complex128)
        for i in range(200):
            # Convert the array of tuples to an array of complex numbers
            print("Converting data to complex numbers: " + str(i+1) + " of 200")
            complex_samples = np.array([s[0] + 1j * s[1] for s in data[i]])

            # Create the complex-valued numpy array
            iq_array = np.array(complex_samples, dtype=np.complex128)

            iq_data[i] = iq_array

        directory = 'dataset/Group' + str(k+1)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        np.savez('dataset/Group' + str(k+1) + '/group' + str(k+1) + '_subset_' + str(j+1) + '.npz', X=iq_data, y=labels)
        # Save the data to a file at the specified path
        print("Saved data to file: " + directory + '/group' + str(k+1) + '_subset_' + str(j+1) + '.npz')
        #print(iq_data[0][:5])
        # Save the data to a file
        #np.savez('data.npz', *data)