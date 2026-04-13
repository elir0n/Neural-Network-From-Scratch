import numpy as np
from .activations import nll_loss


def _one_hot(label, num_classes=10):
    vec = np.zeros((num_classes, 1))
    vec[label] = 1.0
    return vec


def train(model, X_train, y_train, epochs=10, learning_rate=0.001):
    """Train the MLP sample-by-sample using gradient descent.

    Args:
        model: MultilayerPerceptron instance
        X_train: shape (n_samples, 784)
        y_train: integer labels, shape (n_samples,)
        epochs: number of full passes through the training set
        learning_rate: gradient descent step size

    Returns:
        List of average loss values, one per epoch.
    """
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(X_train)):
            X = X_train[i].reshape(-1, 1)
            y = _one_hot(y_train[i])

            Z1, A1, Z2, A2 = model.forward(X)
            total_loss += nll_loss(A2, y)
            model.backward(X, y, Z1, A1, A2, learning_rate)

        avg_loss = total_loss / len(X_train)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}  loss: {avg_loss:.4f}")

    return epoch_losses


def evaluate(model, X_test, y_test):
    """Evaluate the MLP on a test set.

    Args:
        model: trained MultilayerPerceptron instance
        X_test: shape (n_samples, 784)
        y_test: integer labels, shape (n_samples,)

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = 0
    for i in range(len(X_test)):
        X = X_test[i].reshape(-1, 1)
        _, _, _, A2 = model.forward(X)
        pred = np.argmax(A2)
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)
