import os
import cv2
import numpy as np
import torch

# Global cache for ImageNet general object validator
_imagenet_model = None

def _get_imagenet_model():
    global _imagenet_model
    if _imagenet_model is None:
        try:
            import torchvision.models as models
            cache_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth')
            if os.path.exists(cache_path):
                m = models.mobilenet_v2()
                m.load_state_dict(torch.load(cache_path, map_location='cpu'))
                m.eval()
                _imagenet_model = m
        except Exception as e:
            print(f"ImageNet validator initialization notice: {e}")
    return _imagenet_model


def validate_crop_leaf(image_path):
    """
    Validates whether an uploaded image contains an actual crop / plant leaf.
    Filters out human photos/selfies, animals (cats, dogs, birds, etc.), non-plant objects, screenshots, logos.
    Returns: (is_valid: bool, error_message: str or None)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, "Invalid image format. Please upload a clear JPG or PNG image."

        # Standardize for consistent validation
        resized = cv2.resize(image, (320, 320))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)

        r = rgb[:, :, 0].astype(float)
        g = rgb[:, :, 1].astype(float)
        b = rgb[:, :, 2].astype(float)

        # 1. ImageNet General Subject Detection (Animals, People, Objects)
        im_model = _get_imagenet_model()
        if im_model is not None:
            try:
                from torchvision import transforms
                from PIL import Image
                pil_img = Image.open(image_path).convert('RGB')
                t = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor = t(pil_img).unsqueeze(0)
                with torch.no_grad():
                    out = im_model(tensor)
                    probs = torch.softmax(out, dim=1)
                    top_prob, top_idx = torch.topk(probs, 3)
                    
                    # Animals: IDs 0-397 (mammals, pets, birds, etc.)
                    top_id = top_idx[0][0].item()
                    top_p = top_prob[0][0].item() * 100

                    if 0 <= top_id <= 397 and top_p >= 10.0:
                        return False, "An animal / pet was detected in the photo. Please upload a clear photo of a crop or plant leaf."
                    
                    # Man-made items, apparel, human scenes: IDs 400-922
                    if 400 <= top_id <= 922 and top_p >= 25.0:
                        return False, "A non-plant subject was detected. Please upload a clear photo of a crop or plant leaf."
            except Exception as e:
                print(f"ImageNet check warning: {e}")

        # 2. Excess Green Index (ExG)
        exg = 2 * g - r - b
        exg_mask = exg > 10
        exg_ratio = (np.sum(exg_mask) / exg_mask.size) * 100

        # 3. Plant / Leaf Vegetation Tissue
        green_mask = cv2.inRange(hsv, np.array([22, 25, 25]), np.array([98, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([12, 35, 35]), np.array([24, 255, 255]))
        brown_mask = cv2.inRange(hsv, np.array([4, 30, 20]), np.array([18, 255, 200]))

        plant_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(yellow_mask, brown_mask))
        plant_ratio = (np.sum(plant_mask > 0) / plant_mask.size) * 100

        # 4. Human Skin Detection (YCrCb + RGB + HSV Hue <= 18)
        skin_hsv = (hsv[:, :, 0] <= 18) | (hsv[:, :, 0] >= 172)
        skin_ycrcb = (ycrcb[:, :, 1] >= 133) & (ycrcb[:, :, 1] <= 173) & (ycrcb[:, :, 2] >= 77) & (ycrcb[:, :, 2] <= 127)
        skin_rgb = (r > 95) & (g > 40) & (b > 20) & ((np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15) & (np.abs(r - g) > 15) & (r > g) & (r > b)
        skin_mask = skin_hsv & skin_ycrcb & skin_rgb
        skin_ratio = (np.sum(skin_mask) / skin_mask.size) * 100

        # Rule A: Predominant human skin tone over vegetation
        if skin_ratio > 18.0 and skin_ratio > (plant_ratio * 1.5):
            return False, "A personal/portrait photo was detected! Please upload an image of a crop or plant leaf instead."

        # Rule B: Insufficient overall vegetation / plant tissue
        if plant_ratio < 6.0 and exg_ratio < 4.0:
            return False, "No crop or plant leaf was detected in this image. Please upload a clear photo of a crop leaf."

        return True, None
    except Exception as e:
        print(f"Leaf validation warning: {e}")
        return True, None
    except Exception as e:
        print(f"Leaf validation warning: {e}")
        return True, None


def extract_leaf_roi(image_path):
    """
    Extracts the Region of Interest (leaf) from the background.
    Returns a processed numpy image or the original image if extraction fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([22, 25, 25])
        upper_green = np.array([100, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w > 20 and h > 20:
            roi = image[y:y+h, x:x+w]
            return roi
        return image
    except Exception as e:
        print(f"Error in ROI extraction: {e}")
        return cv2.imread(image_path)
