import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def plot_loss_curves(train_losses, val_losses, save_path):
    """Save a training/validation loss curve to disk.

    Args:
        train_losses: list of per-epoch training loss values
        val_losses: list of per-epoch validation loss values
        save_path: file path for the output PNG
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Training loss")
    ax.plot(epochs, val_losses, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def view_classify(img, probs, save_path=None):
    """Visualize an image alongside its predicted class probabilities.

    Args:
        img: tensor of shape (1, 28, 28) or (28, 28)
        probs: 1-D tensor/array of length 10, one probability per class
        save_path: if given, save to disk; otherwise display interactively
    """
    probs = probs.squeeze().cpu().numpy() if hasattr(probs, "cpu") else np.array(probs)
    img_np = img.squeeze().cpu().numpy() if hasattr(img, "cpu") else np.squeeze(img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_np, cmap="gray")
    ax1.axis("off")

    ax2.barh(LABEL_NAMES, probs)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title(f"Predicted: {LABEL_NAMES[int(np.argmax(probs))]}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def random_prediction_example(data_loader, model, save_path=None):
    """Sample one image, run inference, and visualize the result.

    Args:
        data_loader: any DataLoader that yields (images, labels) batches
        model: trained nn.Module
        save_path: if given, save visualization to disk

    Returns:
        predicted_class (int), confidence (float)
    """
    model.eval()
    images, labels = next(iter(data_loader))
    idx = torch.randint(0, images.shape[0], (1,)).item()
    img = images[idx]

    with torch.no_grad():
        log_probs = model(img.unsqueeze(0))
        probs = torch.exp(log_probs).squeeze()

    predicted_class = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_class].item())

    view_classify(img, probs, save_path=save_path)
    return predicted_class, confidence
