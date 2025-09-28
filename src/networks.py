import numpy as np
import src.config as config

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(config.input_size, config.hidden_layer1_size) * np.sqrt(2. / config.input_size)
        self.b1 = np.ones((1, config.hidden_layer1_size))*0.01

        self.W2 = np.random.randn(config.hidden_layer1_size, config.hidden_layer2_size) * np.sqrt(2. / config.hidden_layer1_size)
        self.b2 = np.ones((1, config.hidden_layer2_size))*0.01
        
        self.W3 = np.random.randn(config.hidden_layer2_size, config.num_classes) * np.sqrt(2. / config.hidden_layer2_size)
        self.b3 = np.ones((1, config.num_classes))*0.01

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

        self.dropout_mask1 = None
        self.dropout_mask2 = None
    
    def forward(self, X, y):
        m = X.shape[0]

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.dropout_mask1 = (np.random.rand(*self.a1.shape) > config.dropout_rate)
        self.a1 *= self.dropout_mask1

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.dropout_mask2 = (np.random.rand(*self.a2.shape) > config.dropout_rate)
        self.a2 *= self.dropout_mask2

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)

        loss = -np.mean(np.log(self.a3[np.arange(m), y]))
        return loss
    
    def backward(self, X, y):
        m = X.shape[0]

        grad_z3 = self.a3
        grad_z3[np.arange(m), y] -= 1
        grad_z3 /= m

        grad_W3 = np.dot(self.a2.T, grad_z3)
        grad_b3 = np.sum(grad_z3, axis=0)

        grad_a2 = np.dot(grad_z3, self.W3.T)
        grad_a2 = grad_a2 * self.dropout_mask2
        grad_z2 = grad_a2 * (self.z2 > 0)

        grad_W2 = np.dot(self.a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)

        grad_a1 = np.dot(grad_z2, self.W2.T)
        grad_a1 = grad_a1 * self.dropout_mask1
        grad_z1 = grad_a1 * (self.z1 > 0)

        grad_W1 = np.dot(X.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)

        return grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)