import os
import io
import numpy as np
from PIL import Image
import gdown
import onnxruntime as ort
from flask import Flask, request, send_file, render_template_string

app = Flask(__name__)

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "model.onnx"
GDRIVE_URL = "https://drive.google.com/uc?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading ONNX model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load the ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ------------------------
# IMAGE PROCESSING HELPERS
# ------------------------
def preprocess_image(file_storage, image_size=(256, 256)):
    img = Image.open(file_storage).convert("RGB")
    img = img.resize(image_size)
    data = np.array(img).astype(np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))  # HWC → CHW
    data = np.expand_dims(data, 0)        # add batch dim
    return data

def postprocess_output(output):
    out = output.squeeze(0)               # remove batch
    out = np.transpose(out, (1, 2, 0))    # CHW → HWC
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

# ------------------------
# WEB INTERFACE
# ------------------------
HTML_PAGE = """
<!doctype html>
<title>Face Swap</title>
<h1>Upload two images to swap faces</h1>
<form action="/swap" method="post" enctype="multipart/form-data">
  Source Face: <input type="file" name="src_face" required><br><br>
  Destination Face: <input type="file" name="dst_face" required><br><br>
  <input type="submit" value="Swap Faces">
</form>
{% if result %}
<h2>Result:</h2>
<img src="data:image/png;base64,{{ result }}">
{% endif %}
"""

import base64

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "src_face" not in request.files or "dst_face" not in request.files:
            return "Please upload both images", 400

        src_face = request.files["src_face"]
        dst_face = request.files["dst_face"]

        # preprocess
        src_arr = preprocess_image(src_face)
        dst_arr = preprocess_image(dst_face)

        # Combine or feed separately depending on your model
        # Example: concatenate along channel axis (adjust if your model is different)
        model_input = np.concatenate([src_arr, dst_arr], axis=1)

        # Run inference
        output = session.run(None, {input_name: model_input})[0]

        # Postprocess
        swapped_image = postprocess_output(output)

        # Convert to base64 for inline display
        buf = io.BytesIO()
        swapped_image.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()
        result = base64.b64encode(img_bytes).decode("utf-8")

    return render_template_string(HTML_PAGE, result=result)

# ------------------------
# RUN APP
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT env variable
    app.run(host="0.0.0.0", port=port)