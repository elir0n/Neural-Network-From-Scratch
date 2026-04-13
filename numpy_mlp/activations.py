import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(s):
    return s * (1 - s)


def softmax(z):
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def nll_loss(y_pred, y_true):
    """Negative log-likelihood loss for multiclass classification.

    Args:
        y_pred: softmax probabilities, shape (num_classes, batch_size)
        y_true: one-hot encoded targets, shape (num_classes, batch_size)

    Returns:
        Scalar loss value.
    """
    batch_size = y_true.shape[1]
    log_probs = np.log(y_pred + 1e-8)
    return -np.sum(y_true * log_probs) / batch_size
