"""
app.py
------
Simple Flask web app for the malaria diagnosis support system.
Serves an upload page and runs the trained CNN model on uploaded
blood smear images.

Run:
    python app.py

Then open http://localhost:5000 in a browser.

If model/malaria_resnet50.h5 does not exist yet (you haven't trained
a model), the app still runs in DEMO MODE and returns a clearly-labeled
placeholder prediction, so you can test the web app before training.
"""

import io
import os

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "malaria_resnet50.h5")
IMG_SIZE = (224, 224)

model = None
DEMO_MODE = True

if os.path.exists(MODEL_PATH):
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import preprocess_input

        model = tf.keras.models.load_model(MODEL_PATH)
        DEMO_MODE = False
        print(f"Loaded trained model from {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model ({e}); running in demo mode.")
else:
    print("No trained model found at model/malaria_resnet50.h5 -- running in demo mode.")
    print("Train one with: python train_model.py --data_dir dataset/")


def preprocess_image(image_bytes):
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


@app.route("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    if DEMO_MODE:
        # Placeholder result so the UI can be demoed without a trained model.
        return jsonify({
            "prediction": "negative",
            "confidence": 0.0,
            "demo_mode": True,
            "note": "No trained model loaded. This is a placeholder result. "
                    "Train a model first with train_model.py.",
        })

    processed = preprocess_image(image_bytes)
    raw_score = float(model.predict(processed, verbose=0)[0][0])

    # Class indices from training: {'Parasitized': 0, 'Uninfected': 1}
    # So a LOW score means Parasitized (positive for malaria),
    # and a HIGH score means Uninfected (negative for malaria).
    prediction = "negative" if raw_score > 0.5 else "positive"
    confidence = raw_score if raw_score > 0.5 else 1 - raw_score

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "raw_score": round(raw_score, 4),
        "demo_mode": False,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)