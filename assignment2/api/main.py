from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, json, glob, torch
from torchvision import transforms

from assignment2.helper_lib import get_model
from assignment2.helper_lib.utils import load_model

app = FastAPI(title="Assignment 2 CNN Inference API")

def _find_meta():
    # Allow override
    env = os.getenv("MODEL_META")
    if env and os.path.exists(env):
        return env
    # Common defaults
    for c in [
        "assignment2/checkpoints/cifar_speccnn64.pth.meta.json",
        "assignment2/checkpoints/cnn.pth.meta.json",
    ]:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob("assignment2/checkpoints/*.pth.meta.json")) + \
           sorted(glob.glob("assignment2/checkpoints/*.meta.json"))
    return hits[0] if hits else None

MODEL_META_PATH = _find_meta()
if not MODEL_META_PATH:
    raise RuntimeError("No model meta found. Train first or set MODEL_META env var.")

with open(MODEL_META_PATH, "r") as f:
    meta = json.load(f)

weights_path = meta["weights"]
in_size      = int(meta["in_size"])
in_channels  = int(meta["in_channels"])
classes      = meta["classes"]
model_name   = meta.get("model_name", "CNN")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(model_name, num_classes=len(classes), in_channels=in_channels, in_size=in_size)
load_model(model, weights_path, map_location=device)
model.to(device).eval()

# default to CIFAR-10 stats if RGB/3ch; otherwise neutral 0.5/0.5
if in_channels == 3:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
else:
    mean, std = (0.5,), (0.5,)

preprocess = transforms.Compose([
    transforms.Resize((in_size, in_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "num_classes": len(classes)}

@app.get("/config")
def config():
    preview = classes[:3] + (["..."] if len(classes) > 3 else [])
    return {"meta_path": MODEL_META_PATH, "weights_path": weights_path,
            "model_name": model_name, "in_size": in_size, "in_channels": in_channels,
            "classes_preview": preview, "device": device}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(1, ge=1, le=10)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image input")

    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)

    k = min(top_k, len(classes))
    topk = torch.topk(probs, k=k)
    results = [{"label": classes[i], "prob": float(probs[i])} for i in topk.indices.tolist()]
    return JSONResponse({"predictions": results})
