from torchvision import datasets
from PIL import Image
from pathlib import Path

root = Path("data")
train_dir = root/"train"
test_dir  = root/"test"
(train_dir).mkdir(parents=True, exist_ok=True)
(test_dir).mkdir(parents=True, exist_ok=True)

mnist_train = datasets.MNIST(root="~/.torch-datasets", train=True, download=True)
mnist_test  = datasets.MNIST(root="~/.torch-datasets", train=False, download=True)

def export(split, out_dir):
    for cls in range(10):
        (out_dir/str(cls)).mkdir(parents=True, exist_ok=True)
    for idx, (img, label) in enumerate(split):
        out_path = out_dir/str(label)/f"{idx:06d}.png"
        img.save(out_path)

export(mnist_train, train_dir)
export(mnist_test,  test_dir)
print("Exported MNIST to ImageFolder at ./data/train and ./data/test")
