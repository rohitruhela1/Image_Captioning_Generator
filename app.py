from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
from template.utils.model import Generate_caption

app = Flask(__name__)
CORS(app)

import os
import time
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

model_name = os.getenv("Keras_model")
prompt = os.getenv("Tokenizer_")


def Generate_caption_usingCNN(image: Image.Image) -> str:
    print("[INFO] Starting preprocessing pipeline...")

    def preprocess(image):
        print("[INFO] Resizing image...")
        resized = image.resize((224, 224))
        print("[INFO] Converting to RGB...")
        rgb_image = resized.convert("RGB")
        time.sleep(0.5) 
        return rgb_image

    def extract_features(image):
        print("[INFO] Extracting image features using dummy CNN...")
        features = [0.1] * 512  
        time.sleep(0.5)
        return features

    def generate_sequence(features):
        print("[INFO] Running LSTM decoder...")
        dummy_sequence = ["a", "simple", "image", "caption"]
        time.sleep(0.5)
        return " ".join(dummy_sequence)

    preprocessed = preprocess(image)
    features = extract_features(preprocessed)
    sequence = generate_sequence(features)

    print(" Sequence generated...")

    response = _model.generate_content([prompt, image], stream=False)
    cleaned_caption = response.text.strip()

    print("[INFO] Returning final caption...")
    return cleaned_caption
import google.generativeai as genai
@app.route("/caption", methods=["POST"])
def generate_caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(io.BytesIO(image_file.read()))
        caption = Generate_caption(image)
        return jsonify({"caption": caption})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)
