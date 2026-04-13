import numpy as np
from .activations import sigmoid, sigmoid_derivative, softmax


class MultilayerPerceptron:
    """Two-layer MLP implemented from scratch using NumPy.

    Architecture: input → hidden (sigmoid) → output (softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        """Forward pass for a single sample.

        Args:
            X: input vector, shape (input_size, 1)

        Returns:
            Z1, A1, Z2, A2 — intermediate values needed for backprop
        """
        Z1 = self.W1 @ X + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, y_one_hot, Z1, A1, A2, learning_rate):
        """Backpropagation: compute gradients and update weights in-place.

        Args:
            X: input vector, shape (input_size, 1)
            y_one_hot: one-hot label, shape (output_size, 1)
            Z1, A1, A2: cached activations from forward()
            learning_rate: gradient descent step size
        """
        m = X.shape[1]

        dZ2 = A2 - y_one_hot
        dW2 = dZ2 @ A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = dZ1 @ X.T / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
