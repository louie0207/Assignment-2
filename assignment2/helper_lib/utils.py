import torch
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict

def save_model(model, path: str, extra: Dict[str, Any] | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload["meta"] = extra
    torch.save(payload, path)

def load_model(model, path: str, map_location: str | None = None):
    payload = torch.load(path, map_location=map_location)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(sd)
    return model

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
