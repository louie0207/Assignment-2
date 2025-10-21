FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY assignment2 /app/assignment2
COPY requirements-assignment2.txt /app/requirements-assignment2.txt

# Install PyTorch CPU wheels and deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements-assignment2.txt

# Default to CIFAR-10 SpecCNN64 meta; user can override with -e MODEL_META=...
ENV MODEL_META=assignment2/checkpoints/cifar_speccnn64.pth.meta.json

EXPOSE 8000
CMD ["uvicorn", "assignment2.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
