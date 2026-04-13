import numpy as np
from sklearn.datasets import fetch_openml


def _min_max_norm(X):
    x_min = X.min()
    x_max = X.max()
    return (X - x_min) / (x_max - x_min + 1e-8)


def load_mnist(test_size=0.2):
    """Download and preprocess the MNIST dataset.

    Returns:
        X_train, X_test: float64 arrays of shape (n_samples, 784), normalized to [0, 1]
        y_train, y_test: int arrays of shape (n_samples,), labels 0-9
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    X = _min_max_norm(X)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test
