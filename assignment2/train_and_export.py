import argparse, torch, torch.nn as nn, torch.optim as optim, json
from pathlib import Path
from assignment2.helper_lib import get_data_loader, get_model, train_model, evaluate_model, save_model, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-train", default="data/train")
    ap.add_argument("--data-test",  default="data/test")
    ap.add_argument("--in-size", type=int, default=32)
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--model", default="CNN", choices=["FCNN","CNN","EnhancedCNN","resnet18"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="assignment2/checkpoints/cnn.pth")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = get_data_loader(args.data_train, batch_size=args.batch_size, train=True,
                                   in_size=args.in_size, in_channels=args.in_channels)
    test_loader  = get_data_loader(args.data_test,  batch_size=args.batch_size, train=False,
                                   in_size=args.in_size, in_channels=args.in_channels)

    # derive classes from the train ImageFolder
    classes = train_loader.dataset.classes
    num_classes = len(classes)

    model = get_model(args.model, num_classes=num_classes, in_channels=args.in_channels, in_size=args.in_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=args.epochs)
    avg_loss, acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"[Test] loss={avg_loss:.4f} acc={acc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out, extra={
        "in_size": args.in_size,
        "in_channels": args.in_channels,
        "classes": classes,
        "model_name": args.model
    })
    # handy sidecar for API
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "weights": args.out,
        "in_size": args.in_size,
        "in_channels": args.in_channels,
        "classes": classes,
        "model_name": args.model
    }, indent=2))
    print(f"Saved: {args.out} and {meta_path}")

if __name__ == "__main__":
    main()
