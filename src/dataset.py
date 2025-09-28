import os
import cv2 as cv
import numpy as np
import src.config as config

def load_training_data():
    print("Loading training data...")
    
    # Define the path to the training data directory
    data_path = os.path.join(config.data_path, "train")

    X_list = []
    y_list = []
    labels = {'cats': 0, 'dogs': 1, 'snakes': 2}

    for img_name in os.listdir(data_path):
        try:
            # Construct the full image path
            img_path = os.path.join(data_path, img_name)

            # Get the label from the filename and append to y_list
            label_name = img_name.split('_')[0]
            y_list.append(labels[label_name])

            # Read and convert the image
            img = cv.resize(cv.imread(img_path), (config.input_dim, config.input_dim))
            if img is not None:
                # Convert from BGR (default for OpenCV) to RGB
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                X_list.append(img)
            else:
                print(f"Warning: Could not read image at {img_path}")
        except IndexError:
            print(f"Warning: Skipping file with invalid name format: {img_name}")
    
    # Convert the lists to NumPy arrays after the loop
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Reshape the array and re-assign the result
        X_reshaped = X.reshape(len(X), -1)
        print(f"Loaded {X_reshaped.shape[0]} training samples.")

        return X_reshaped/255, y
    else:
        # Return empty arrays if no images were loaded
        print("No training data found.")
        
        return np.array([]), np.array([])

def load_testing_data():
    print("Loading testing data...")
    
    # Define the path to the testing data directory
    data_path = os.path.join(config.data_path, "test")

    X_list = []
    y_list = []
    labels = {'cats': 0, 'dogs': 1, 'snakes': 2}

    for img_name in os.listdir(data_path):
        try:
            # Construct the full image path
            img_path = os.path.join(data_path, img_name)

            # Get the label from the filename and append to y_list
            label_name = img_name.split('_')[0]
            y_list.append(labels[label_name])

            # Read and convert the image
            img = cv.resize(cv.imread(img_path), (config.input_dim, config.input_dim))
            if img is not None:
                # Convert from BGR (default for OpenCV) to RGB
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                X_list.append(img)
            else:
                print(f"Warning: Could not read image at {img_path}")
        except IndexError:
            print(f"Warning: Skipping file with invalid name format: {img_name}")
    
    # Convert the lists to NumPy arrays after the loop
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Reshape the array and re-assign the result
        X_reshaped = X.reshape(len(X), -1)
        print(f"Loaded {X_reshaped.shape[0]} testing samples.")

        return X_reshaped/255, y
    else:
        # Return empty arrays if no images were loaded
        print("No testing data found.")

        return np.array([]), np.array([])