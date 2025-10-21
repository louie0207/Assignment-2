import torch
from torch import nn

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, data_loader, criterion: nn.Module, device: str = "cpu"):
    model.to(device)
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total += targets.size(0)
    avg_loss = total_loss / max(total, 1)
    accuracy = total_correct / max(total, 1)
    return avg_loss, accuracy
