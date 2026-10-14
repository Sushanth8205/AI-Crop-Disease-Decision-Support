import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from app.ml.cnn_model import CropDiseaseModel
from app.ml.image_utils import extract_leaf_roi, validate_crop_leaf


# Mapping of class index to disease name.
# Keep this order exactly aligned with your model's training labels.
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# Models are loaded only when the first prediction is requested.
device = torch.device("cpu")
ensemble_models = None
model = None
_model_load_lock = threading.Lock()


def _build_model_architecture(model_type, num_classes):
    """Build the architecture corresponding to the saved checkpoint."""

    model_type_lower = model_type.lower()

    if "best_model" in model_type_lower or "efficientnet" in model_type_lower:
        network = models.efficientnet_b0(weights=None)
        num_features = network.classifier[1].in_features
        network.classifier[1] = nn.Linear(num_features, num_classes)

    elif "mobilenet" in model_type_lower:
        network = models.mobilenet_v2(weights=None)
        network.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(network.last_channel, num_classes)
        )

    else:
        network = CropDiseaseModel(num_classes=num_classes)

    return network


def _load_single_model(name, filename, num_classes, target_device):
    """Load a trained PyTorch checkpoint."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, filename)

    if not os.path.isfile(checkpoint_path):
        print(f"Model checkpoint not found: {checkpoint_path}")
        return None

    try:
        print(f"Loading model '{name}' from {filename}...")

        network = _build_model_architecture(filename, num_classes)

        # Load the checkpoint onto CPU first to reduce device-related issues.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )
        except TypeError:
            # Compatibility with older PyTorch versions.
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu"
            )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state dictionary.")

        # Remove common prefixes added during training.
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]

            if key.startswith("model."):
                key = key[len("model."):]

            clean_state_dict[key] = value

        network.load_state_dict(clean_state_dict, strict=True)
        network.to(target_device)
        network.eval()

        print(f"Successfully loaded '{name}' from {filename}")
        return network

    except Exception as exc:
        print(f"Failed to load '{name}' from {filename}: {exc}")
        return None


def load_ensemble_models():
    """Load the available trained model and return models plus device."""

    target_device = torch.device("cpu")
    num_classes = len(CLASS_NAMES)

    # Only use the checkpoint that you intend to deploy.
    candidates = [
        ("MobileNetV2", "mobilenetv2_plant.pth"),
    ]

    loaded = []

    for name, filename in candidates:
        loaded_model = _load_single_model(
            name,
            filename,
            num_classes,
            target_device
        )

        if loaded_model is not None:
            loaded.append((name, loaded_model))

    # Do not silently substitute a different architecture unless its
    # checkpoint is known to match that architecture and class ordering.
    if loaded:
        print(f"Loaded trained model(s): {[name for name, _ in loaded]}")
    else:
        print("ERROR: No trained model could be loaded.")

    return loaded, target_device


def _get_models():
    """Load the model once, even if multiple requests arrive together."""

    global ensemble_models, model, device

    if ensemble_models is None:
        with _model_load_lock:
            if ensemble_models is None:
                loaded_models, loaded_device = load_ensemble_models()
                ensemble_models = loaded_models
                device = loaded_device
                model = ensemble_models[0][1] if ensemble_models else None

    return ensemble_models, device


def predict_disease(image_path):
    """
    Predict crop disease from an uploaded image.

    Returns:
        (crop_name, disease_name, confidence_percentage, error_message)
    """

    # 1. Validate the uploaded image.
    try:
        is_valid, error_msg = validate_crop_leaf(image_path)
    except Exception:
        print("Image validation failed:")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, (
            "Unable to validate this image. Please upload a clear crop leaf photo."
        )

    if not is_valid:
        return None, None, 0.0, error_msg

    # 2. Load the trained model on demand.
    loaded_models, target_device = _get_models()

    if not loaded_models:
        return None, None, 0.0, (
            "The trained disease detection model could not be loaded. "
            "Please try again later."
        )

    # 3. Extract the leaf region and open the image.
    try:
        roi_image = extract_leaf_roi(image_path)

        if roi_image is None:
            pil_image = Image.open(image_path).convert("RGB")
        else:
            # The ROI utility is assumed to return an OpenCV BGR image.
            roi_rgb = roi_image[:, :, ::-1].copy()
            pil_image = Image.fromarray(roi_rgb).convert("RGB")

    except Exception:
        print("Image processing failed:")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, (
            "Unable to process this image. Please upload a valid image file."
        )

    # 4. Prepare the image for MobileNetV2.
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    try:
        image_tensor = image_transform(pil_image).unsqueeze(0).to(target_device)

        # 5. Run trained model inference.
        with torch.inference_mode():
            probabilities = []

            for model_name, trained_model in loaded_models:
                outputs = trained_model(image_tensor)

                if outputs.ndim != 2 or outputs.shape[1] != len(CLASS_NAMES):
                    raise ValueError(
                        f"{model_name} returned {outputs.shape[1]} classes; "
                        f"expected {len(CLASS_NAMES)}."
                    )

                probabilities.append(F.softmax(outputs, dim=1))

            combined_probabilities = torch.stack(probabilities).mean(dim=0)
            confidence_tensor, predicted_index = torch.max(
                combined_probabilities, dim=1
            )

            confidence = round(confidence_tensor.item() * 100, 2)
            predicted_class = CLASS_NAMES[predicted_index.item()]

    except Exception:
        print("Disease prediction failed:")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, (
            "The disease prediction failed. Please try another clear leaf image."
        )

    # 6. Parse the predicted class.
    if "___" not in predicted_class:
        return None, None, confidence, (
            "The model returned an unrecognized disease label."
        )

    crop_label, disease_label = predicted_class.split("___", 1)
    crop_name = crop_label.replace("_", " ").replace(",", "")
    disease_name = disease_label.replace("_", " ").strip()

    return crop_name, disease_name, confidence, None
