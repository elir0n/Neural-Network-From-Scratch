# Neural Network From Scratch

Two multilayer perceptrons built and trained end-to-end in Python — one written purely in NumPy to expose every matrix operation and gradient update, and one written in PyTorch to show how a modern framework handles the same ideas at scale.

---

## Experiments

### Part 1 — NumPy MLP on MNIST

Classifies handwritten digits (0–9) using a two-layer network implemented with no ML framework. Every forward pass, activation function, and backpropagation step is written by hand using NumPy.

```
Input (784)  →  Hidden (128, sigmoid)  →  Output (10, softmax)
```

- Dataset: MNIST — 70,000 grayscale images (28×28 px), 80/20 train/test split
- Loss: Negative log-likelihood
- Optimizer: Vanilla gradient descent, sample-by-sample
- Hyperparameters: `hidden=128`, `lr=0.001`, `epochs=10`
- Result: ~88% test accuracy

### Part 2 — PyTorch MLP on Fashion-MNIST

Classifies clothing items (T-shirt, Sneaker, Bag, etc.) using a three-layer network built with PyTorch's `nn.Module`. Uses mini-batch SGD with train/validation tracking and produces a loss curve.

```
Input (784)  →  FC(128, ReLU)  →  FC(64, ReLU)  →  Output (10, LogSoftmax)
```

- Dataset: Fashion-MNIST — 60,000 grayscale images (28×28 px), 80/20 train/val split
- Loss: NLLLoss (paired with LogSoftmax output)
- Optimizer: SGD, `lr=0.005`, `batch_size=64`
- Hyperparameters: `epochs=5`
- Result: ~85% validation accuracy

---

## Project Structure

```
Neural-Network-From-Scratch/
├── main.py                  # Entry point — runs one or both experiments
├── requirements.txt
│
├── numpy_mlp/
│   ├── activations.py       # sigmoid, softmax, nll_loss
│   ├── data.py              # load_mnist()
│   ├── model.py             # MultilayerPerceptron (forward + backward)
│   └── train.py             # train(), evaluate()
│
├── pytorch_mlp/
│   ├── data.py              # load_fashion_mnist()
│   ├── model.py             # NeuralNetwork(nn.Module)
│   ├── train.py             # train_model(), evaluate_model()
│   └── visualize.py         # plot_loss_curves(), random_prediction_example()
│
└── outputs/                 # Auto-created — loss curves and prediction plots
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run both experiments

```bash
python main.py
```

### 3. Run a single experiment

```bash
python main.py --part 1   # NumPy MLP on MNIST only
python main.py --part 2   # PyTorch MLP on Fashion-MNIST only
```

Datasets are downloaded automatically on first run. Output figures are saved to `outputs/`.

---

## Key Concepts

| Concept | Where it appears |
|---|---|
| Min-max normalization | `numpy_mlp/data.py` |
| Sigmoid + derivative | `numpy_mlp/activations.py` |
| Softmax with numerical stability | `numpy_mlp/activations.py` |
| Manual backpropagation (chain rule) | `numpy_mlp/model.py` |
| `nn.Module` and `forward()` | `pytorch_mlp/model.py` |
| Mini-batch SGD with DataLoader | `pytorch_mlp/train.py` |
| Train/validation split | `pytorch_mlp/data.py` |
| Loss curve visualization | `pytorch_mlp/visualize.py` |