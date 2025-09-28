import math
import numpy as np
import src.config as config
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.loss_history = []

    def update_parameters(self, grads, learning_rate):
        grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3 = grads

        self.neural_net.W1 -= learning_rate * grad_W1
        self.neural_net.b1 -= learning_rate * grad_b1
        self.neural_net.W2 -= learning_rate * grad_W2
        self.neural_net.b2 -= learning_rate * grad_b2
        self.neural_net.W3 -= learning_rate * grad_W3
        self.neural_net.b3 -= learning_rate * grad_b3

    def train(self, X, y):
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X = (X - mean_X) / std_X

        print(f'Epochs 0/{config.num_epochs} completed, Loss: {0:.4f}')
        for epoch in range(config.num_epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            loss_epoch = 0
            for i in range(0, X.shape[0], config.batch_size):
                X_batch = X_shuffled[i:i+config.batch_size]
                y_batch = y_shuffled[i:i+config.batch_size]

                loss = self.neural_net.forward(X_batch, y_batch)
                loss_epoch += loss

                grads = self.neural_net.backward(X_batch, y_batch)
                self.update_parameters(grads, config.learning_rate)
                self.loss_history.append(loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epochs {epoch+1}/{config.num_epochs} completed, Loss: {loss_epoch/10:.4f}')


