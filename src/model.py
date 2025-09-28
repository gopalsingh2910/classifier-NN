import numpy as np
from src.networks import NeuralNetwork
from src.train import TrainModel
from src.utils import stats
import src.config as config

class Model:
    def __init__(self):
        self.nn = NeuralNetwork()
    
    def train(self, X, y):
        print(f'Training model...')
        trainer = TrainModel(self.nn)
        trainer.train(X, y)
        print('Training completed.')

        print(f'Plotting stats...')
        stats(np.arange(config.runs), np.array(trainer.loss_history))
        print(f'Plotting stats completed.')
        return
    
    def predict(self, X):
        z1 = np.dot(X, self.nn.W1) + self.nn.b1
        a1 = self.nn.relu(z1)*(1-config.dropout_rate)

        z2 = np.dot(a1, self.nn.W2) + self.nn.b2
        a2 = self.nn.relu(z2)*(1-config.dropout_rate)

        z3 = np.dot(a2, self.nn.W3) + self.nn.b3
        a3 = self.nn.softmax(z3)

        y = np.argmax(a3, axis=1)
        return y

    def test(self, X, y):
        print(f'Testing model...')
        
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        X = (X - mean_x) / std_x
        
        predictions = self.predict(X)
        print(f'Testing completed.')

        accuracy = np.mean(predictions == y)
        return accuracy

