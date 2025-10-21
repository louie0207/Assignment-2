FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY assignment2 /app/assignment2
COPY requirements-assignment2.txt /app/requirements-assignment2.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# CPU-only torch/torchvision
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements-assignment2.txt

# ✅ Bake weights + meta into the image
# (make sure the two files exist locally before build)
#   assignment2/checkpoints/cifar_speccnn64.pth
#   assignment2/checkpoints/cifar_speccnn64.pth.meta.json
# If they’re already inside "assignment2", this is already covered by the earlier COPY.
# Otherwise uncomment next line:
# COPY assignment2/checkpoints /app/assignment2/checkpoints

ENV MODEL_META=assignment2/checkpoints/cifar_speccnn64.pth.meta.json

EXPOSE 8000
CMD ["uvicorn", "assignment2.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
