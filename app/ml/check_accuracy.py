import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ml.cnn_model import CropDiseaseModel


def init_and_load_model(model_name, model_path, num_classes, device):
    """Initializes model architecture and loads weights from checkpoint."""
    print(f"Loading '{model_name}' from: {model_path}")
    
    if "best_model" in model_path.lower() or "efficientnet" in model_name.lower():
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif "mobilenet" in model_path.lower() or "mobilenet" in model_name.lower():
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes)
        )
    else:
        model = CropDiseaseModel(num_classes=num_classes)

    model = model.to(device)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Clean 'model.' prefix if present
            clean_state_dict = {
                (k.replace('model.', '') if k.startswith('model.') else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(clean_state_dict)
            print(f"  -> Successfully loaded weights for {model_name}")
        except Exception as e:
            print(f"  -> Warning: Could not load weights for {model_name}: {e}")
    else:
        print(f"  -> Warning: File not found at {model_path}. Using uninitialized weights.")

    model.eval()
    return model


def compute_metrics(y_true, y_pred, latencies):
    """Calculates accuracy, precision, recall, f1, rmse, and avg latency."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "accuracy": accuracy * 100.0,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "rmse": rmse,
        "avg_latency": avg_latency
    }


def evaluate_model(data_dir="knowledge_base/plantvillage dataset/color", max_samples=200):
    """
    Evaluates both EfficientNet-B0 and MobileNetV2 individually AND combined (ensemble)
    in a single run across the same test dataset.
    """
    print("====================================================================")
    print("        ML QUALITY & PERFORMANCE COMBINED EVALUATION               ")
    print("====================================================================")

    if not os.path.exists(data_dir):
        print(f"Error: Test dataset directory '{data_dir}' not found.")
        print("Please provide a valid dataset path to run the evaluation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Total dataset size: {len(full_dataset)} images across {num_classes} classes")

    # Split dataset deterministically into 80% train, 20% test (SAME SEED AS TRAIN)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    sample_count = min(max_samples, len(test_dataset))
    print(f"Evaluating on {sample_count} test samples (batch size = 1 for per-image latency)\n")

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Locate models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_configs = [
        {
            "name": "EfficientNet-B0",
            "path": os.path.join(current_dir, "best_model.pth")
        },
        {
            "name": "MobileNetV2",
            "path": os.path.join(current_dir, "mobilenetv2_plant.pth")
        }
    ]

    loaded_models = {}
    for cfg in model_configs:
        loaded_models[cfg["name"]] = init_and_load_model(
            cfg["name"], cfg["path"], num_classes, device
        )

    # Containers for results
    y_true = []
    predictions = {
        "EfficientNet-B0": [],
        "MobileNetV2": [],
        "Combined (Ensemble)": []
    }
    latencies = {
        "EfficientNet-B0": [],
        "MobileNetV2": [],
        "Combined (Ensemble)": []
    }

    print("\nRunning inference evaluation...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= sample_count:
                break

            inputs = inputs.to(device)
            true_label = labels.item()
            y_true.append(true_label)

            # Model 1: EfficientNet-B0
            t0 = time.perf_counter()
            out_eff = loaded_models["EfficientNet-B0"](inputs)
            t1 = time.perf_counter()
            lat_eff = (t1 - t0) * 1000
            prob_eff = F.softmax(out_eff, dim=1)
            pred_eff = torch.argmax(prob_eff, dim=1).item()

            predictions["EfficientNet-B0"].append(pred_eff)
            latencies["EfficientNet-B0"].append(lat_eff)

            # Model 2: MobileNetV2
            t2 = time.perf_counter()
            out_mob = loaded_models["MobileNetV2"](inputs)
            t3 = time.perf_counter()
            lat_mob = (t3 - t2) * 1000
            prob_mob = F.softmax(out_mob, dim=1)
            pred_mob = torch.argmax(prob_mob, dim=1).item()

            predictions["MobileNetV2"].append(pred_mob)
            latencies["MobileNetV2"].append(lat_mob)

            # Combined Ensemble (Softmax averaging)
            prob_comb = (prob_eff + prob_mob) / 2.0
            pred_comb = torch.argmax(prob_comb, dim=1).item()
            predictions["Combined (Ensemble)"].append(pred_comb)
            latencies["Combined (Ensemble)"].append(lat_eff + lat_mob)

    # Compute metrics for each
    metrics_results = {}
    for name in ["EfficientNet-B0", "MobileNetV2", "Combined (Ensemble)"]:
        metrics_results[name] = compute_metrics(
            y_true, predictions[name], latencies[name]
        )

    # Print Detailed Outputs
    print("\n" + "=" * 68)
    print("                     EVALUATION RESULTS SUMMARY                      ")
    print("=" * 68)
    print(f"{'Model / Pipeline':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Latency':<10}")
    print("-" * 68)

    for name, m in metrics_results.items():
        print(f"{name:<22} | {m['accuracy']:>7.2f}% | {m['precision']:>7.2f}% | {m['recall']:>7.2f}% | {m['f1']:>7.2f}% | {m['avg_latency']:>7.2f} ms")

    print("=" * 68)
    print("\nRequirements Verification:")
    for name, m in metrics_results.items():
        status = "[PASS]" if m['avg_latency'] < 200 else "[FAIL]"
        print(f" - {name:<22}: {status} Latency requirement (< 200 ms): {m['avg_latency']:.2f} ms")
    print("=" * 68)

    return metrics_results


if __name__ == "__main__":
    evaluate_accuracy = evaluate_model  # For backwards compatibility
    evaluate_model()
