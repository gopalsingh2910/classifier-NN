import os

dropout_rate = 0.5
batch_size = 100
num_epochs = 100

learning_rate = 0.005
hidden_layer1_size = 64
hidden_layer2_size = 32

input_dim = 128
num_classes = 3
input_size = input_dim * input_dim * 3

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
runs = num_epochs*len(os.listdir(os.path.join(data_path, "train")))//batch_size