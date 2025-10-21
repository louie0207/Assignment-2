import argparse, json
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from assignment2.helper_lib import get_model, train_model, evaluate_model, save_model, set_seed

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SpecCNN64")
    ap.add_argument("--in-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="assignment2/checkpoints/cifar_speccnn64.pth",
                    help="Path to save weights (meta saved as <out>.meta.json)")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm_train = transforms.Compose([
        transforms.Resize((args.in_size, args.in_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize((args.in_size, args.in_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root="~/.torch-datasets", train=True,  download=True, transform=tfm_train)
    test_set  = datasets.CIFAR10(root="~/.torch-datasets", train=False, download=True, transform=tfm_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    classes      = train_set.classes
    num_classes  = len(classes)
    in_channels  = 3

    model = get_model(args.model, num_classes=num_classes, in_channels=in_channels, in_size=args.in_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=args.epochs)
    avg_loss, acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"[Test] loss={avg_loss:.4f} acc={acc:.4f}")

    # Save weights and meta
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out, extra={
        "in_size": args.in_size, "in_channels": in_channels, "classes": classes, "model_name": args.model
    })
    meta_path = Path(args.out).with_suffix(".pth.meta.json")
    meta = {"weights": args.out, "in_size": args.in_size, "in_channels": in_channels, "classes": classes, "model_name": args.model}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {args.out} and {meta_path}")

if __name__ == "__main__":
    main()
