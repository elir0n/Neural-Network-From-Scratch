import torch


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=5):
    """Train a PyTorch model and track train/validation loss per epoch.

    Args:
        model: nn.Module instance
        optimizer: torch optimizer
        criterion: loss function (expects log-probabilities, e.g. NLLLoss)
        train_loader: DataLoader for training batches
        val_loader: DataLoader for validation batches
        epochs: number of training epochs

    Returns:
        train_losses, val_losses: lists of average loss per epoch
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                loss = criterion(output, labels)
                running_val_loss += loss.item()

        avg_train = running_train_loss / len(train_loader)
        avg_val = running_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"  Epoch {epoch + 1}/{epochs}  train loss: {avg_train:.4f}  val loss: {avg_val:.4f}")

    return train_losses, val_losses


def evaluate_model(model, val_loader):
    """Compute classification accuracy on a validation/test set.

    Args:
        model: trained nn.Module
        val_loader: DataLoader for evaluation

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            log_probs = model(images)
            probs = torch.exp(log_probs)
            _, predicted = probs.topk(1, dim=1)
            correct += predicted.squeeze().eq(labels).sum().item()
            total += labels.size(0)
    return correct / total
