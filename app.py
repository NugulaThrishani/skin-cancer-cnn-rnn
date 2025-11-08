import os
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Use TensorFlow's Keras to load the .keras model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    load_model = None
    print("Warning: TensorFlow not available at import time:", e)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(APP_ROOT, "Melanoma-003.keras"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
CLASSES = os.environ.get("CLASSES", "benign,melanoma").split(",")

app = Flask(__name__)
model = None


def preprocess_image(file_stream, target_size=IMAGE_SIZE):
    """Open image from file stream, convert to RGB, resize, normalize and return batch array."""
    image = Image.open(file_stream).convert("RGB")
    image = image.resize((target_size, target_size))
    arr = np.asarray(image).astype("float32") / 255.0
    # ensure shape (1, H, W, C)
    return np.expand_dims(arr, axis=0)


def load_model_safe(path):
    global model
    if load_model is None:
        raise RuntimeError("TensorFlow not installed or failed to import")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = load_model(path)
    print(f"Model loaded from {path}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        try:
            load_model_safe(MODEL_PATH)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file part in the request with key 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img_batch = preprocess_image(file.stream, IMAGE_SIZE)
        preds = model.predict(img_batch)
        # handle common output shapes: (1,1) for single-sigmoid, (1,N) for multi-class
        preds_list = preds.flatten().tolist()

        # Convert probabilities to a readable dict
        if preds.shape[-1] == 1 or len(preds_list) == 1:
            prob = float(preds_list[0])
            # assume binary: [prob_of_positive]
            class_name = CLASSES[1] if prob >= 0.5 and len(CLASSES) > 1 else CLASSES[0]
            result = {
                "predictions": {CLASSES[0]: round(1 - prob, 6), CLASSES[1] if len(CLASSES) > 1 else "positive": round(prob, 6)},
                "predicted_class": class_name,
                "raw": preds_list,
            }
        else:
            # multiclass softmax-like outputs
            # map classes if provided; otherwise use indices
            probs = [float(p) for p in preds_list]
            if len(CLASSES) == len(probs):
                mapping = {CLASSES[i]: round(probs[i], 6) for i in range(len(probs))}
                pred_idx = int(np.argmax(probs))
                pred_class = CLASSES[pred_idx]
            else:
                mapping = {str(i): round(probs[i], 6) for i in range(len(probs))}
                pred_idx = int(np.argmax(probs))
                pred_class = str(pred_idx)

            result = {
                "predictions": mapping,
                "predicted_class": pred_class,
                "raw": preds_list,
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Try loading model at start for fast failure
    try:
        load_model_safe(MODEL_PATH)
    except Exception as e:
        print("Model not loaded at startup:", e)
        print("You can set MODEL_PATH env var or place 'model.keras' in project root.")

    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
