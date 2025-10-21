import torch
import torch.nn as nn
from torchvision.models import resnet18

class FCNN(nn.Module):
    def __init__(self, in_hw=32, num_classes=10, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_hw * in_hw * in_channels, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.features(x); x = torch.flatten(x, 1); return self.classifier(x)

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# === EXACT Part-1 spec ===
class SpecCNN64(nn.Module):
    """
    64×64×3 → Conv(3→16,3,1,1)+ReLU → MaxPool(2)
            → Conv(16→32,3,1,1)+ReLU → MaxPool(2)
            → Flatten → FC(100)+ReLU → FC(10)
    """
    def __init__(self, num_classes=10, in_channels=3, in_size=64):
        super().__init__()
        assert in_size == 64 and in_channels == 3
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 64→32
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 32→16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100), nn.ReLU(),
            nn.Linear(100, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

def _tweak_resnet_small_inputs(m: nn.Module, remove_maxpool: bool = True):
    if remove_maxpool and hasattr(m, "maxpool"):
        m.maxpool = nn.Identity()
    if hasattr(m, "conv1") and isinstance(m.conv1, nn.Conv2d) and m.conv1.stride == (2, 2):
        m.conv1.stride = (1, 1)
    return m

def get_model(model_name: str, num_classes: int = 10, in_channels: int = 1, in_size: int = 32):
    name = model_name.lower()
    if name == "fcnn":         return FCNN(in_hw=in_size, num_classes=num_classes, in_channels=in_channels)
    if name == "cnn":          return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "enhancedcnn":  return EnhancedCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "speccnn64":    return SpecCNN64(num_classes=num_classes, in_channels=in_channels, in_size=in_size)
    if name == "resnet18":
        m = resnet18(weights=None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return _tweak_resnet_small_inputs(m, remove_maxpool=(in_size <= 64))
    raise ValueError(f"Unknown model_name: {model_name}")
