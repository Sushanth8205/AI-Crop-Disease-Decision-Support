import os
import tempfile

from flask import Flask, jsonify, request
from app.ml.inference import predict_disease

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

API_KEY = os.environ.get("INFERENCE_API_KEY", "")


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    if not API_KEY:
        return jsonify({"error": "API key is not configured"}), 503

    supplied_key = request.headers.get("X-API-Key", "")
    if supplied_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({"error": "Please upload an image using the image field"}), 400

    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "No image selected"}), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
            uploaded.save(temp_path)

        crop, disease, confidence, error = predict_disease(temp_path)

        if error:
            return jsonify({"error": str(error)}), 422

        return jsonify({
            "crop": crop,
            "disease": disease,
            "confidence": float(confidence)
        }), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed on the inference server"}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image must be 10 MB or smaller"}), 413
