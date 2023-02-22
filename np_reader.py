import os
import numpy as np

# Define the number of groups and subsets
num_groups = 4
num_subsets = 50

# Loop over each group
for k in range(num_groups):
    print("Group: " + str(k+1) + " of " + str(num_groups))
    
    # Loop over each subset
    for j in range(num_subsets):
        # Define the file path
        file_path = 'dataset/Group' + str(k+1) + '/group' + str(k+1) + '_subset_' + str(j+1) + '.npz'
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the data from the file
            data = np.load(file_path)
            iq_data = data['X']
            label_data = data['y']
            
            # Print the shape of the data arrays
            print("Subset " + str(j+1) + " of " + str(num_subsets) + ":")
            print("  iq_data shape: " + str(iq_data.shape))
            print("  label_data shape: " + str(label_data.shape))
        else:
            print("Subset " + str(j+1) + " of " + str(num_subsets) + " not found.")
