
import cv2
import numpy as np


def validate_crop_leaf(image_path):
    """
    Lightweight image validation using OpenCV only.
    Returns (is_valid, error_message).
    Does not load a second neural network.
    """
    try:
        image = cv2.imread(image_path)

        if image is None:
            return False, (
                "Invalid image format. Please upload a clear JPG or PNG image."
            )

        # Resize before processing to limit memory use.
        resized = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Detect green, yellow, and some brown plant tissue.
        green_mask = cv2.inRange(
            hsv,
            np.array([22, 25, 25], dtype=np.uint8),
            np.array([98, 255, 255], dtype=np.uint8)
        )

        yellow_mask = cv2.inRange(
            hsv,
            np.array([12, 35, 35], dtype=np.uint8),
            np.array([24, 255, 255], dtype=np.uint8)
        )

        brown_mask = cv2.inRange(
            hsv,
            np.array([4, 30, 20], dtype=np.uint8),
            np.array([18, 255, 200], dtype=np.uint8)
        )

        plant_mask = cv2.bitwise_or(
            green_mask,
            cv2.bitwise_or(yellow_mask, brown_mask)
        )

        plant_ratio = (
            np.count_nonzero(plant_mask) / plant_mask.size
        ) * 100

        # Excess Green Index helps detect green vegetation.
        b, g, r = cv2.split(resized)
        exg = (
            2.0 * g.astype(np.float32)
            - r.astype(np.float32)
            - b.astype(np.float32)
        )

        exg_ratio = (
            np.count_nonzero(exg > 10) / exg.size
        ) * 100

        if plant_ratio < 6.0 and exg_ratio < 4.0:
            return False, (
                "No clear plant or leaf tissue was detected. "
                "Please upload a clear photo of a crop leaf."
            )

        return True, None

    except Exception:
        print("Leaf validation failed:")
        import traceback
        traceback.print_exc()

        return False, (
            "Unable to validate this image. Please upload a valid crop leaf photo."
        )


def extract_leaf_roi(image_path):
    """
    Extract a likely green leaf region using OpenCV.
    Returns a BGR image, or None if the image cannot be read.
    """
    try:
        image = cv2.imread(image_path)

        if image is None:
            return None

        # Work on a smaller image to reduce temporary allocations.
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))

        if scale < 1.0:
            small = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            small = image

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([22, 25, 25], dtype=np.uint8),
            np.array([100, 255, 255], dtype=np.uint8)
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w <= 20 or h <= 20:
            return image

        # Convert coordinates back to the original image dimensions.
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        height, width = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return image

        return roi

    except Exception:
        print("Leaf region extraction failed:")
        import traceback
        traceback.print_exc()

        return None
