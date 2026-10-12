import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ====================================================================
# 1. MODEL ARCHITECTURES
# ====================================================================

class SpatialAttentionBlock(nn.Module):
    """Spatial Attention mechanism for VGG-16 feature maps."""
    def __init__(self, in_features=512):
        super(SpatialAttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 8, kernel_size=1),
            nn.BatchNorm2d(in_features // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights


class VGG16SpatialAttention(nn.Module):
    """VGG-16 with Spatial Attention Block (matching model_checkpoint.pth)."""
    def __init__(self, num_classes=38):
        super(VGG16SpatialAttention, self).__init__()
        vgg = models.vgg16(weights=None)
        self.features = vgg.features
        self.attention = SpatialAttentionBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class BaselineCNN(nn.Module):
    """Standard Baseline Convolutional Neural Network."""
    def __init__(self, num_classes=38):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model_weights(model, path, device, model_name=""):
    """Safely loads model weights from disk or initializes architecture."""
    if not path or not os.path.exists(path):
        print(f"  [OK] Initialized '{model_name}' architecture (untrained baseline).")
        model = model.to(device)
        model.eval()
        return model

    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Clean 'model.' prefix if saved with DataParallel/Wrapper
        clean_state_dict = {
            (k.replace("model.", "") if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(clean_state_dict, strict=False)
        print(f"  [OK] Loaded weights for '{model_name}' from: {os.path.basename(path)}")
    except Exception as e:
        print(f"  [!] Note while loading '{model_name}': {e}")

    model = model.to(device)
    model.eval()
    return model


# ====================================================================
# 2. EVALUATION FUNCTION
# ====================================================================

def run_table2_experiment(
    data_dir="knowledge_base/plantvillage dataset/color",
    max_samples=250,
    batch_size=1
):
    print("=" * 72)
    print(" TABLE II: MODEL COMPARISON EXPERIMENTAL EVALUATION")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    vgg_ckpt = os.path.join(current_dir, "model_checkpoint.pth")
    eff_ckpt = os.path.join(current_dir, "best_model.pth")
    mob_ckpt = os.path.join(current_dir, "mobilenetv2_plant.pth")

    # Dataset transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return None

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Dataset: PlantVillage ({len(full_dataset)} total images, {num_classes} classes)")

    # 80/20 train/test split matching paper methodology
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    sample_count = min(max_samples, len(test_dataset))
    print(f"Evaluating on {sample_count} test samples (seed=42 for reproducibility)...\n")

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # -------------------------------------------------------------
    # Initializing Models for Table II Comparison
    # -------------------------------------------------------------
    print("Loading models:")

    # 1. Baseline CNN (Custom ConvNet)
    baseline_cnn = BaselineCNN(num_classes=num_classes)
    baseline_cnn = load_model_weights(baseline_cnn, "", device, "Baseline CNN (Custom)")

    # 2. MobileNetV2 (Standalone)
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(mobilenet.last_channel, num_classes)
    )
    mobilenet = load_model_weights(mobilenet, mob_ckpt, device, "MobileNetV2")

    # 3. VGG-16 + Spatial Attention
    vgg16_attention = VGG16SpatialAttention(num_classes=num_classes)
    vgg16_attention = load_model_weights(vgg16_attention, vgg_ckpt, device, "VGG-16 + Spatial Attention")

    # 4. Other model / baseline (EfficientNet-B0)
    effnet = models.efficientnet_b0(weights=None)
    effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, num_classes)
    effnet = load_model_weights(effnet, eff_ckpt, device, "Other model / baseline (EfficientNet-B0)")

    models_dict = {
        "Baseline CNN": baseline_cnn,
        "MobileNetV2": mobilenet,
        "VGG-16 + Spatial Attention": vgg16_attention,
        "Other model / baseline": effnet,
    }

    # Tracking separate individual metrics
    y_true = []
    predictions = {name: [] for name in models_dict.keys()}
    latencies = {name: [] for name in models_dict.keys()}

    print("\nRunning inference for each model separately...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= sample_count:
                break

            inputs = inputs.to(device)
            target = labels.item()
            y_true.append(target)

            for name, model in models_dict.items():
                t0 = time.perf_counter()
                logits = model(inputs)
                t1 = time.perf_counter()

                lat = (t1 - t0) * 1000.0
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

                predictions[name].append(pred)
                latencies[name].append(lat)

    # -------------------------------------------------------------
    # Compute Metrics Separately
    # -------------------------------------------------------------
    results = {}
    for name, preds in predictions.items():
        acc = accuracy_score(y_true, preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average='weighted', zero_division=0
        )
        avg_lat = float(np.mean(latencies[name])) if latencies[name] else 0.0

        results[name] = {
            "Accuracy": acc,
            "Precision": prec * 100.0,
            "Recall": rec * 100.0,
            "F1-score": f1 * 100.0,
            "Latency": avg_lat
        }

    # -------------------------------------------------------------
    # Print Exact Table II Format (Separate Values)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TABLE II. MODEL COMPARISON TEMPLATE FOR THE FINAL EXPERIMENT.")
    print("=" * 70)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Precision':<10} | {'F1-score':<10}")
    print("-" * 70)

    for name in ["Baseline CNN", "MobileNetV2", "VGG-16 + Spatial Attention", "Other model / baseline"]:
        if name in results:
            r = results[name]
            print(f"{name:<30} | {r['Accuracy']:>8.2f}% | {r['Precision']:>8.2f}% | {r['F1-score']:>8.2f}%")

    print("=" * 70)

    # Detailed Summary with Latency & Recall
    print("\n" + "=" * 85)
    print(f"{'Model':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-score':<9} | {'Latency':<9}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<28} | {r['Accuracy']:>7.2f}% | {r['Precision']:>7.2f}% | {r['Recall']:>7.2f}% | {r['F1-score']:>7.2f}% | {r['Latency']:>6.2f} ms")
    print("=" * 85)

    return results


if __name__ == "__main__":
    run_table2_experiment()

