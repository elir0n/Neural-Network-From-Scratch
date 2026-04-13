"""
Neural Network From Scratch
============================
Entry point that runs one or both experiments:

  Part 1 — NumPy MLP trained on MNIST (handwritten digits)
  Part 2 — PyTorch MLP trained on Fashion-MNIST (clothing items)

Usage:
  python main.py            # run both experiments
  python main.py --part 1   # NumPy experiment only
  python main.py --part 2   # PyTorch experiment only
"""

import argparse
import os


OUTPUTS_DIR = "outputs"


def run_numpy_experiment():
    from numpy_mlp.data import load_mnist
    from numpy_mlp.model import MultilayerPerceptron
    from numpy_mlp.train import train, evaluate

    print("\n" + "=" * 60)
    print("PART 1 — NumPy MLP on MNIST")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_mnist()

    model = MultilayerPerceptron(input_size=784, hidden_size=128, output_size=10)

    print("\nTraining...")
    losses = train(model, X_train, y_train, epochs=10, learning_rate=0.001)

    print("\nEvaluating...")
    accuracy = evaluate(model, X_test, y_test)

    print(f"\nResult — Test accuracy: {accuracy * 100:.2f}%")
    return accuracy


def run_pytorch_experiment():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from pytorch_mlp.data import load_fashion_mnist
    from pytorch_mlp.model import NeuralNetwork
    from pytorch_mlp.train import train_model, evaluate_model
    from pytorch_mlp.visualize import plot_loss_curves, random_prediction_example

    print("\n" + "=" * 60)
    print("PART 2 — PyTorch MLP on Fashion-MNIST")
    print("=" * 60)

    train_loader, val_loader = load_fashion_mnist(data_path="./data")

    model = NeuralNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.005)
    criterion = nn.NLLLoss()

    print("\nTraining...")
    train_losses, val_losses = train_model(
        model, optimizer, criterion, train_loader, val_loader, epochs=5
    )

    print("\nEvaluating...")
    accuracy = evaluate_model(model, val_loader)

    loss_path = os.path.join(OUTPUTS_DIR, "loss_curves.png")
    plot_loss_curves(train_losses, val_losses, save_path=loss_path)
    print(f"  Loss curve saved to {loss_path}")

    pred_path = os.path.join(OUTPUTS_DIR, "sample_prediction.png")
    predicted_class, confidence = random_prediction_example(
        val_loader, model, save_path=pred_path
    )
    from pytorch_mlp.visualize import LABEL_NAMES
    print(f"  Sample prediction saved to {pred_path}")
    print(f"  Predicted: {LABEL_NAMES[predicted_class]} ({confidence * 100:.1f}% confidence)")

    print(f"\nResult — Validation accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Neural Network From Scratch")
    parser.add_argument(
        "--part",
        choices=["1", "2", "all"],
        default="all",
        help="Which experiment to run: 1 (NumPy/MNIST), 2 (PyTorch/Fashion-MNIST), or all",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    results = {}
    if args.part in ("1", "all"):
        results["numpy_mnist"] = run_numpy_experiment()
    if args.part in ("2", "all"):
        results["pytorch_fashion"] = run_pytorch_experiment()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "numpy_mnist" in results:
        print(f"  NumPy MLP  — MNIST test accuracy:          {results['numpy_mnist'] * 100:.2f}%")
    if "pytorch_fashion" in results:
        print(f"  PyTorch MLP — Fashion-MNIST val accuracy:  {results['pytorch_fashion'] * 100:.2f}%")
    print()


if __name__ == "__main__":
    main()
