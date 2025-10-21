# Assignment 2 – CNN

Implements the Part-1 spec CNN and deploys a CIFAR-10 classifier via FastAPI.

- **Part 1:** Exact architecture (64×64 RGB): Conv16 → ReLU → MaxPool → Conv32 → ReLU → MaxPool → Flatten → FC100 → ReLU → FC10 (see SpecCNN64).
- **Part 2:** Train on CIFAR-10 and expose /predict API (Docker optional).
- **Part 3:** Theory/calculation answers are included in Assignment2.pdf.

---

## Requirements

- Python 3.12 (macOS/Linux)
- (Optional) Docker
- (Optional) Virtual environment (e.g., `.venv-assignment2`)

```bash
# (optional) create/activate a venv
python -m pip install -r requirements-assignment2.txt

# Project Layout
assignment2/
  api/
    main.py
  helper_lib/
    __init__.py
    data_loader.py
    model.py
    trainer.py
    evaluator.py
    utils.py
  train_cifar10.py
  checkpoints/
    # weights + sidecar meta go here after training
  Dockerfile
requirements-assignment2.txt
Assignment2.pdf

# 1) Train on CIFAR-10 (SpecCNN64 @ 64×64)
SpecCNN64 matches the assignment’s architecture exactly (RGB 64×64 input).
python -m assignment2.train_cifar10 \
  --model SpecCNN64 \
  --in-size 64 \
  --batch-size 128 \
  --epochs 10 \
  --out assignment2/checkpoints/cifar_speccnn64.pth

## This produces:
assignment2/checkpoints/cifar_speccnn64.pth
assignment2/checkpoints/cifar_speccnn64.pth.meta.json

## The meta JSON includes:
{
  "weights": "assignment2/checkpoints/cifar_speccnn64.pth",
  "in_size": 64,
  "in_channels": 3,
  "classes": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"],
  "model_name": "SpecCNN64"
}

## Normalization used (train & API):
CIFAR-10 mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616).

# 2) Run the API (local)
uvicorn assignment2.api.main:app --reload
## Swagger UI
## http://127.0.0.1:8000/docs

## Quick checks:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/config

## Predict (single image)
curl -s -X POST "http://127.0.0.1:8000/predict?top_k=1" \
  -F "file=@/absolute/path/to/image.jpg;type=image/jpeg" | python -m json.tool

## Model discovery order:
MODEL_META env var (if set)
assignment2/checkpoints/cifar_speccnn64.pth.meta.json
assignment2/checkpoints/cnn.pth.meta.json
First *.meta.json found in assignment2/checkpoints/
## Override explicitly if needed:
export MODEL_META=assignment2/checkpoints/cifar_speccnn64.pth.meta.json
uvicorn assignment2.api.main:app --reload

# 3) Docker (CPU)
docker build -t assign2-cnn -f assignment2/Dockerfile .
# Mount checkpoints so you (or the grader) can swap models if desired
docker run --rm -p 8000:8000 \
  -e MODEL_META=assignment2/checkpoints/cifar_speccnn64.pth.meta.json \
  -v "$(pwd)/assignment2/checkpoints:/app/assignment2/checkpoints:ro" \
  assign2-cnn

# (Optional) Local sample images for testing
## If you want a few CIFAR-10 test images saved to disk:
from torchvision import datasets, transforms
from PIL import Image
ds = datasets.CIFAR10("~/.torch-datasets", train=False, download=True, transform=transforms.ToTensor())
for i in range(5):
    img, y = ds[i]
    cls = ds.classes[y]
    Image.fromarray((img.permute(1,2,0).numpy()*255).astype("uint8")).save(f"/tmp/{i}_{cls}.jpg","JPEG")
print("Saved to /tmp")
## Then upload any of those via Swagger /predict.

# Tips / Troubleshooting
Consistency: API preprocessing uses the same CIFAR-10 normalization as training.
GPU/CPU: Script auto-selects CUDA if available; otherwise CPU.
Overwriting checkpoints: Re-running training with the same --out path overwrites the .pth and refreshes the .pth.meta.json.
Classes preview: GET /config shows a small preview of class names to verify you’re using CIFAR-10.

# (Optional) MNIST note
This project focuses on CIFAR-10 for grading. If you maintain any MNIST files, keep them outside the main path (e.g., extras/) to avoid confusion and ensure the API defaults to the CIFAR meta.