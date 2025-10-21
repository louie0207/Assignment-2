import torch
from torch import nn
from tqdm import tqdm
from typing import Optional

def train_model(
    model: torch.nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 10,
    grad_clip_norm: Optional[float] = None,
):
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        tqdm.write(f"[Train] Epoch {epoch}: loss={epoch_loss:.4f} acc={epoch_acc:.4f}")
    return model
