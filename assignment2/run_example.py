import torch
import torch.nn as nn
import torch.optim as optim
from assignment2.helper_lib import (
    get_data_loader, get_model, train_model,
    evaluate_model, save_model, set_seed
)

def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_size = 32
    in_channels = 1      # set to 3 if your images are RGB
    num_classes = 10

    train_loader = get_data_loader(
        'data/train', batch_size=64, train=True,
        in_size=in_size, in_channels=in_channels, num_workers=2
    )
    test_loader  = get_data_loader(
        'data/test', batch_size=64, train=False,
        in_size=in_size, in_channels=in_channels, num_workers=2
    )

    model = get_model("CNN", num_classes=num_classes, in_channels=in_channels, in_size=in_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=5)
    avg_loss, acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"Test loss: {avg_loss:.4f} | Test acc: {acc:.4f}")

    save_model(model, "assignment2/checkpoints/cnn.pth",
               extra={"in_size": in_size, "in_channels": in_channels, "num_classes": num_classes})

if __name__ == "__main__":
    main()
