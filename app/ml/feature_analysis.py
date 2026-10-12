import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib
# Use Agg backend if non-interactive, but allow show if display exists
if os.environ.get('DISPLAY', '') == '' and sys.platform != 'win32':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ml.image_utils import extract_leaf_roi
from app.ml.inference import CLASS_NAMES


def get_feature_extractor_model(device):
    """Loads feature extractor (MobileNetV2 or EfficientNet-B0)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mobilenet_path = os.path.join(current_dir, "mobilenetv2_plant.pth")
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, len(CLASS_NAMES))
    )

    if os.path.exists(mobilenet_path):
        checkpoint = torch.load(mobilenet_path, map_location=device, weights_only=False)
        sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()})
        print(f"Loaded trained feature extractor weights from {os.path.basename(mobilenet_path)}")
    else:
        print("Using base architecture for feature analysis.")

    model = model.to(device)
    model.eval()
    return model


def visualize_preprocessing_and_features(image_path, output_dir=None, show_plot=True):
    """
    Performs end-to-end Preprocessing & Feature Extraction analysis
    and generates comprehensive visual diagnostic plots.
    """
    print("====================================================================")
    print("        VISUAL PREPROCESSING & FEATURE EXTRACTION ANALYSIS          ")
    print("====================================================================")

    if not os.path.exists(image_path):
        print(f"Error: Sample image '{image_path}' not found.")
        return

    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # STEP 1: PREPROCESSING STAGES
    # -------------------------------------------------------------
    print("\n[Stage 1] Preprocessing Input Image...")
    # 1.1 Original Image
    raw_bgr = cv2.imread(image_path)
    if raw_bgr is None:
        print(f"Error reading image at {image_path}")
        return
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = raw_rgb.shape
    print(f" - Raw Image Dimensions : {w}x{h} ({c} channels, {w * h * c:,} pixels)")

    # 1.2 Leaf ROI & HSV Mask
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_bgr = raw_bgr.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest_contour)
        cv2.rectangle(roi_bgr, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
        cropped_roi = raw_rgb[ry:ry + rh, rx:rx + rw]
    else:
        cropped_roi = raw_rgb

    # 1.3 Transform to 224x224 Tensor & Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_img = Image.fromarray(raw_rgb)
    tensor_input = transform(pil_img).unsqueeze(0).to(device)
    print(f" - Normalized Tensor     : {list(tensor_input.shape)} (Standardized mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")

    # -------------------------------------------------------------
    # STEP 2: FEATURE EXTRACTION & ACTIVATION MAPS
    # -------------------------------------------------------------
    print("\n[Stage 2] Deep Convolutional Feature Extraction...")
    model = get_feature_extractor_model(device)

    # Capture early, mid, and final conv layer activations
    activations = {}
    def hook_fn(name):
        def hook(mod, inp, out):
            activations[name] = out.detach()
        return hook

    # Register forward hooks on MobileNet features
    # Early conv (layer 0 or 1), mid conv (layer 6), deep conv (layer 18)
    if hasattr(model, 'features'):
        model.features[0].register_forward_hook(hook_fn('early_conv'))
        model.features[6].register_forward_hook(hook_fn('mid_conv'))
        model.features[18].register_forward_hook(hook_fn('deep_conv'))

    with torch.no_grad():
        features_raw = model.features(tensor_input)
        pooled = nn.AdaptiveAvgPool2d((1, 1))(features_raw)
        feature_vector = torch.flatten(pooled, 1).cpu().numpy().flatten()

    print(f" - Raw Deep Feature Maps : {list(features_raw.shape)} ({features_raw.shape[1]} channels x {features_raw.shape[2]}x{features_raw.shape[3]})")
    print(f" - Condensed Fingerprint : {feature_vector.shape[0]} dimensional dense embedding vector")

    # Compute Feature Heatmap (Activation Attention Map)
    deep_act = activations.get('deep_conv', features_raw).cpu().squeeze(0) # (1280, 7, 7)
    heatmap = torch.mean(deep_act, dim=0).numpy() # (7, 7)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(raw_rgb, 0.6, heatmap_colored_rgb, 0.4, 0)

    # -------------------------------------------------------------
    # STEP 3: BUILD VISUAL REPORT FIGURE
    # -------------------------------------------------------------
    print("\n[Stage 3] Generating Visual Dashboard...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12), facecolor='#f8fafc')
    fig.suptitle(f"Crop Disease Feature Analysis & Preprocessing Pipeline\nSource: {os.path.basename(image_path)}", fontsize=16, fontweight='bold', color='#1e293b', y=0.98)

    # Subplot 1: Raw Original Image
    ax1 = plt.subplot2grid((3, 4), (0, 0))
    ax1.imshow(raw_rgb)
    ax1.set_title(f"1. Original RGB\n({w}x{h} px)", fontsize=11, fontweight='semibold')
    ax1.axis('off')

    # Subplot 2: HSV Leaf Segmentation Mask
    ax2 = plt.subplot2grid((3, 4), (0, 1))
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("2. Leaf Segmentation\n(HSV Green Mask)", fontsize=11, fontweight='semibold')
    ax2.axis('off')

    # Subplot 3: ROI Bounding Box Crop
    ax3 = plt.subplot2grid((3, 4), (0, 2))
    ax3.imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    ax3.set_title("3. ROI Extraction\n(Contour Bounding Box)", fontsize=11, fontweight='semibold')
    ax3.axis('off')

    # Subplot 4: Color Channel Distribution (Histogram)
    ax4 = plt.subplot2grid((3, 4), (0, 3))
    colors = ('#ef4444', '#22c55e', '#3b82f6')
    for idx, col in enumerate(colors):
        hist = cv2.calcHist([raw_rgb], [idx], None, [256], [0, 256])
        ax4.plot(hist, color=col, lw=1.5, label=['Red', 'Green', 'Blue'][idx])
    ax4.set_title("4. Color Channel Distribution", fontsize=11, fontweight='semibold')
    ax4.set_xlim([0, 256])
    ax4.set_xlabel("Pixel Intensity (0-255)", fontsize=9)
    ax4.set_ylabel("Pixel Count", fontsize=9)
    ax4.legend(loc='upper right', fontsize=8)

    # Subplot 5: Early Layer Feature Filters (Edges / Veins)
    ax5 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
    early_maps = activations.get('early_conv', None)
    if early_maps is not None:
        early_np = early_maps.cpu().squeeze(0).numpy() # (32, 112, 112)
        # Create 4x4 grid of 16 filters
        grid_rows, grid_cols = 2, 8
        grid_img = np.zeros((grid_rows * early_np.shape[1], grid_cols * early_np.shape[2]))
        for idx in range(min(16, early_np.shape[0])):
            r, c_idx = divmod(idx, grid_cols)
            f_slice = early_np[idx]
            f_norm = (f_slice - f_slice.min()) / (f_slice.max() - f_slice.min() + 1e-8)
            grid_img[r * early_np.shape[1]:(r + 1) * early_np.shape[1], c_idx * early_np.shape[2]:(c_idx + 1) * early_np.shape[2]] = f_norm
        ax5.imshow(grid_img, cmap='viridis')
    ax5.set_title("5. Early Conv Feature Maps (Edges, Texture & Leaf Veins)", fontsize=11, fontweight='semibold')
    ax5.axis('off')

    # Subplot 6: Deep Convolutional Feature Maps
    ax6 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
    deep_maps = activations.get('deep_conv', features_raw)
    deep_np = deep_maps.cpu().squeeze(0).numpy() # (1280, 7, 7)
    grid_rows, grid_cols = 4, 8
    grid_deep = np.zeros((grid_rows * deep_np.shape[1], grid_cols * deep_np.shape[2]))
    for idx in range(min(32, deep_np.shape[0])):
        r, c_idx = divmod(idx, grid_cols)
        f_slice = deep_np[idx]
        f_norm = (f_slice - f_slice.min()) / (f_slice.max() - f_slice.min() + 1e-8)
        grid_deep[r * deep_np.shape[1]:(r + 1) * deep_np.shape[1], c_idx * deep_np.shape[2]:(c_idx + 1) * deep_np.shape[2]] = f_norm
    ax6.imshow(grid_deep, cmap='inferno')
    ax6.set_title(f"6. Deep Layer Feature Activations (Sample of 32 / {deep_np.shape[0]} Filters)", fontsize=11, fontweight='semibold')
    ax6.axis('off')

    # Subplot 7: Spatial Attention / Disease Heatmap Overlay
    ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    ax7.imshow(overlay)
    ax7.set_title("7. Disease Attention Heatmap Overlay (Feature Focus Zones)", fontsize=11, fontweight='semibold')
    ax7.axis('off')

    # Subplot 8: Condensed Feature Vector Fingerprint
    ax8 = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    ax8.plot(feature_vector, color='#0284c7', lw=0.8)
    ax8.fill_between(range(len(feature_vector)), feature_vector, color='#38bdf8', alpha=0.3)
    ax8.set_title(f"8. Condensed Disease Fingerprint Vector ({len(feature_vector)} Dense Dimensions)", fontsize=11, fontweight='semibold')
    ax8.set_xlabel("Feature Dimension Index (0 - 1279)", fontsize=9)
    ax8.set_ylabel("Activation Intensity", fontsize=9)
    ax8.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save visualization to file
    save_path = os.path.join(output_dir, "feature_analysis_report.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n[Saved Report] Visual analysis saved to: {save_path}")

    # Also save a web-accessible copy in app/static/ if it exists
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
    if os.path.exists(static_dir):
        static_save_path = os.path.join(static_dir, "feature_analysis_report.png")
        fig.savefig(static_save_path, dpi=200, bbox_inches='tight')
        print(f"[Saved Report] Static copy saved for UI to: {static_save_path}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    print("====================================================================")


def analyze_features(image_path):
    """Entry point preserving existing function signature."""
    visualize_preprocessing_and_features(image_path, show_plot=True)


if __name__ == "__main__":
    # Pick a sample image from the PlantVillage dataset
    dataset_path = "knowledge_base/plantvillage dataset/color"
    sample_img_path = None

    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_img_path = os.path.join(root, file)
                    break
            if sample_img_path:
                break

    if sample_img_path:
        print(f"Analyzing sample: {sample_img_path}")
        analyze_features(sample_img_path)
    else:
        print(f"Could not find any images in '{dataset_path}'. Please specify an image path.")
