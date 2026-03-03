import io
import os
import cv2
import base64
import numpy as np
from flask import Flask, request, send_file, jsonify
import insightface
from insightface.app import FaceAnalysis
import traceback

app_flask = Flask(__name__)
face_app = None
swapper = None
models_loaded = False

def init_models():
    global face_app, swapper, models_loaded
    if models_loaded:
        return True
    try:
        if face_app is None:
            print("Loading face analysis model...")
            face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            print("Face analysis model loaded.")
        if swapper is None:
            print("Loading swapper model...")
            model_path = os.path.join("models", "inswapper_128.onnx")
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")
                return False
            swapper = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
            print("Swapper model loaded.")
        models_loaded = True
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

def read_image_from_request(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def resize_image(img, max_size=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

def extract_face_crop(img, face, padding=30, target_size=150):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (target_size, target_size))
    return face_crop

def create_faces_grid(faces_list, face_size=150):
    if len(faces_list) == 0:
        blank = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        cv2.putText(blank, "No faces", (10, face_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return blank
    
    n = len(faces_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    
    grid_h = rows * face_size
    grid_w = cols * face_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i, face_img in enumerate(faces_list):
        row = i // cols
        col = i % cols
        y = row * face_size
        x = col * face_size
        grid[y:y+face_size, x:x+face_size] = face_img
        cv2.putText(grid, str(i), (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return grid

@app_flask.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models_loaded}), 200

@app_flask.route("/", methods=["GET"])
def index():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Face Swap</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          * { box-sizing: border-box; }
          body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
          h2 { color: #333; text-align: center; }
          .container { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
          .step { margin-bottom: 25px; }
          .step-title { font-size: 18px; font-weight: bold; color: #007bff; margin-bottom: 15px; }
          .upload-row { display: flex; gap: 20px; flex-wrap: wrap; }
          .upload-box { flex: 1; min-width: 200px; }
          .upload-box label { display: block; margin-bottom: 8px; font-weight: bold; color: #555; }
          input[type="file"] { width: 100%; padding: 10px; border: 2px dashed #ccc; border-radius: 8px; background: #fafafa; }
          button { background: #007bff; color: white; border: none; padding: 12px 25px; border-radius: 8px; cursor: pointer; font-size: 16px; margin-top: 15px; }
          button:hover { background: #0056b3; }
          button:disabled { background: #ccc; cursor: not-allowed; }
          .faces-container { display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }
          .faces-section { flex: 1; min-width: 250px; }
          .faces-section h4 { margin: 0 0 10px 0; padding: 10px; border-radius: 8px; color: white; }
          .source-header { background: #28a745; }
          .target-header { background: #dc3545; }
          .faces-grid { display: flex; flex-wrap: wrap; gap: 10px; min-height: 100px; background: #f8f9fa; padding: 15px; border-radius: 8px; }
          .face-item { text-align: center; cursor: pointer; border: 3px solid transparent; border-radius: 8px; padding: 5px; transition: all 0.2s; }
          .face-item:hover { border-color: #007bff; }
          .face-item.selected { border-color: #ffc107; background: #fff3cd; }
          .face-item img { width: 80px; height: 80px; object-fit: cover; border-radius: 5px; }
          .face-item .label { font-size: 12px; font-weight: bold; margin-top: 5px; }
          .swap-pairs { margin-top: 15px; padding: 15px; background: #e9ecef; border-radius: 8px; }
          .swap-pair { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; padding: 8px; background: white; border-radius: 5px; }
          .swap-pair img { width: 50px; height: 50px; object-fit: cover; border-radius: 5px; }
          .arrow { font-size: 20px; color: #007bff; }
          .remove-pair { background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; }
          #resultContainer { margin-top: 20px; text-align: center; }
          #resultImage { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
          .loading { display: none; text-align: center; padding: 20px; }
          .loading.active { display: block; }
          .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
          @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
          .status { padding: 10px; border-radius: 5px; margin-top: 10px; }
          .status.success { background: #d4edda; color: #155724; }
          .status.error { background: #f8d7da; color: #721c24; }
        </style>
      </head>
      <body>
        <h2>Face Swap Tool</h2>
        
        <div class="container">
          <div class="step">
            <div class="step-title">Step 1: Upload Images</div>
            <div class="upload-row">
              <div class="upload-box">
                <label>Source Image (face to use)</label>
                <input type="file" id="sourceFile" accept="image/*">
              </div>
              <div class="upload-box">
                <label>Target Image (image to modify)</label>
                <input type="file" id="targetFile" accept="image/*">
              </div>
            </div>
            <button onclick="detectFaces()" id="detectBtn">Detect Faces</button>
            <div class="loading" id="detectLoading"><div class="spinner"></div><p>Detecting faces...</p></div>
          </div>
        </div>

        <div class="container" id="facesContainer" style="display:none;">
          <div class="step">
            <div class="step-title">Step 2: Select Faces to Swap</div>
            <p>Click a SOURCE face, then click a TARGET face to create a swap pair.</p>
            <div class="faces-container">
              <div class="faces-section">
                <h4 class="source-header">SOURCE Faces</h4>
                <div class="faces-grid" id="sourceFaces"></div>
              </div>
              <div class="faces-section">
                <h4 class="target-header">TARGET Faces</h4>
                <div class="faces-grid" id="targetFaces"></div>
              </div>
            </div>
            
            <div class="swap-pairs" id="swapPairs">
              <strong>Swap Pairs:</strong>
              <div id="pairsList"><em>No pairs yet. Click source face then target face.</em></div>
            </div>
            
            <button onclick="performSwap()" id="swapBtn" disabled>Perform Swap</button>
            <div class="loading" id="swapLoading"><div class="spinner"></div><p>Swapping faces...</p></div>
          </div>
        </div>

        <div class="container" id="resultContainer" style="display:none;">
          <div class="step-title">Result</div>
          <img id="resultImage" src="">
          <br><br>
          <a id="downloadLink" href="" download="swapped.jpg"><button>Download Result</button></a>
        </div>

        <div id="statusMsg"></div>

        <script>
          let sourceFacesData = [];
          let targetFacesData = [];
          let swapPairs = [];
          let selectedSourceIdx = null;
          let sourceFileGlobal = null;
          let targetFileGlobal = null;

          function showStatus(msg, isError) {
            const el = document.getElementById('statusMsg');
            el.innerHTML = '<div class="status ' + (isError ? 'error' : 'success') + '">' + msg + '</div>';
            setTimeout(() => el.innerHTML = '', 5000);
          }

          async function detectFaces() {
            const sourceFile = document.getElementById('sourceFile').files[0];
            const targetFile = document.getElementById('targetFile').files[0];
            
            if (!sourceFile || !targetFile) {
              showStatus('Please select both source and target images.', true);
              return;
            }

            sourceFileGlobal = sourceFile;
            targetFileGlobal = targetFile;

            document.getElementById('detectBtn').disabled = true;
            document.getElementById('detectLoading').classList.add('active');

            const formData = new FormData();
            formData.append('source', sourceFile);
            formData.append('target', targetFile);

            try {
              const response = await fetch('/detect_faces', { method: 'POST', body: formData });
              const data = await response.json();

              if (data.error) {
                showStatus(data.error, true);
                return;
              }

              sourceFacesData = data.faces.filter(f => f.type === 'source');
              targetFacesData = data.faces.filter(f => f.type === 'target');

              renderFaces();
              document.getElementById('facesContainer').style.display = 'block';
              swapPairs = [];
              selectedSourceIdx = null;
              renderPairs();
              showStatus('Detected ' + sourceFacesData.length + ' source faces and ' + targetFacesData.length + ' target faces.', false);

            } catch (e) {
              showStatus('Error detecting faces: ' + e.message, true);
            } finally {
              document.getElementById('detectBtn').disabled = false;
              document.getElementById('detectLoading').classList.remove('active');
            }
          }

          function renderFaces() {
            const srcContainer = document.getElementById('sourceFaces');
            const tgtContainer = document.getElementById('targetFaces');
            srcContainer.innerHTML = '';
            tgtContainer.innerHTML = '';

            sourceFacesData.forEach((face, i) => {
              const div = document.createElement('div');
              div.className = 'face-item';
              div.dataset.type = 'source';
              div.dataset.index = face.index;
              div.innerHTML = '<img src="data:image/jpeg;base64,' + face.image + '"><div class="label">S' + face.index + '</div>';
              div.onclick = () => selectFace('source', face.index, div);
              srcContainer.appendChild(div);
            });

            targetFacesData.forEach((face, i) => {
              const div = document.createElement('div');
              div.className = 'face-item';
              div.dataset.type = 'target';
              div.dataset.index = face.index;
              div.innerHTML = '<img src="data:image/jpeg;base64,' + face.image + '"><div class="label">T' + face.index + '</div>';
              div.onclick = () => selectFace('target', face.index, div);
              tgtContainer.appendChild(div);
            });
          }

          function selectFace(type, index, el) {
            if (type === 'source') {
              document.querySelectorAll('#sourceFaces .face-item').forEach(e => e.classList.remove('selected'));
              el.classList.add('selected');
              selectedSourceIdx = index;
            } else if (type === 'target' && selectedSourceIdx !== null) {
              const existing = swapPairs.find(p => p.target_index === index);
              if (existing) {
                showStatus('Target face ' + index + ' already has a swap pair.', true);
                return;
              }
              swapPairs.push({ source_index: selectedSourceIdx, target_index: index });
              selectedSourceIdx = null;
              document.querySelectorAll('#sourceFaces .face-item').forEach(e => e.classList.remove('selected'));
              renderPairs();
            } else {
              showStatus('Select a SOURCE face first.', true);
            }
          }

          function renderPairs() {
            const container = document.getElementById('pairsList');
            if (swapPairs.length === 0) {
              container.innerHTML = '<em>No pairs yet. Click source face then target face.</em>';
              document.getElementById('swapBtn').disabled = true;
              return;
            }

            container.innerHTML = '';
            swapPairs.forEach((pair, i) => {
              const srcFace = sourceFacesData.find(f => f.index === pair.source_index);
              const tgtFace = targetFacesData.find(f => f.index === pair.target_index);
              const div = document.createElement('div');
              div.className = 'swap-pair';
              div.innerHTML = 
                '<img src="data:image/jpeg;base64,' + srcFace.image + '">' +
                '<span>S' + pair.source_index + '</span>' +
                '<span class="arrow">→</span>' +
                '<img src="data:image/jpeg;base64,' + tgtFace.image + '">' +
                '<span>T' + pair.target_index + '</span>' +
                '<button class="remove-pair" onclick="removePair(' + i + ')">X</button>';
              container.appendChild(div);
            });
            document.getElementById('swapBtn').disabled = false;
          }

          function removePair(index) {
            swapPairs.splice(index, 1);
            renderPairs();
          }

          async function performSwap() {
            if (swapPairs.length === 0) {
              showStatus('Add at least one swap pair.', true);
              return;
            }

            document.getElementById('swapBtn').disabled = true;
            document.getElementById('swapLoading').classList.add('active');

            const formData = new FormData();
            formData.append('source', sourceFileGlobal);
            formData.append('target', targetFileGlobal);
            formData.append('swaps', JSON.stringify(swapPairs));

            try {
              const response = await fetch('/swap_selected', { method: 'POST', body: formData });
              
              if (!response.ok) {
                const err = await response.json();
                showStatus(err.error || 'Swap failed', true);
                return;
              }

              const blob = await response.blob();
              const url = URL.createObjectURL(blob);
              
              document.getElementById('resultImage').src = url;
              document.getElementById('downloadLink').href = url;
              document.getElementById('resultContainer').style.display = 'block';
              showStatus('Swap completed successfully!', false);

            } catch (e) {
              showStatus('Error performing swap: ' + e.message, true);
            } finally {
              document.getElementById('swapBtn').disabled = false;
              document.getElementById('swapLoading').classList.remove('active');
            }
          }
        </script>
      </body>
    </html>
    """

@app_flask.route("/detect_source", methods=["POST"])
def detect_source():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files:
            return jsonify({"error": "Missing 'source' file."}), 400

        src_img = read_image_from_request(request.files["source"])
        if src_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        src_faces = face_app.get(src_img)

        faces_data = []
        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                faces_data.append({
                    "index": i,
                    "image": b64_str
                })

        return jsonify({
            "count": len(faces_data),
            "faces": faces_data
        })
    except Exception as e:
        print(f"Error in detect_source: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect_target", methods=["POST"])
def detect_target():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "target" not in request.files:
            return jsonify({"error": "Missing 'target' file."}), 400

        tgt_img = read_image_from_request(request.files["target"])
        if tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        tgt_img = resize_image(tgt_img)
        tgt_faces = face_app.get(tgt_img)

        face_crops = []
        for face in tgt_faces:
            crop = extract_face_crop(tgt_img, face)
            face_crops.append(crop)

        grid = create_faces_grid(face_crops)
        ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode image."}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="target_faces.jpg"
        )
    except Exception as e:
        print(f"Error in detect_target: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect", methods=["POST"])
def detect_both():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        src_crops = [extract_face_crop(src_img, f) for f in src_faces]
        tgt_crops = [extract_face_crop(tgt_img, f) for f in tgt_faces]

        src_grid = create_faces_grid(src_crops)
        tgt_grid = create_faces_grid(tgt_crops)

        src_h, src_w = src_grid.shape[:2]
        tgt_h, tgt_w = tgt_grid.shape[:2]

        label_h = 30
        max_w = max(src_w, tgt_w)
        
        src_grid_padded = np.zeros((src_h, max_w, 3), dtype=np.uint8)
        src_grid_padded[:, :src_w] = src_grid
        
        tgt_grid_padded = np.zeros((tgt_h, max_w, 3), dtype=np.uint8)
        tgt_grid_padded[:, :tgt_w] = tgt_grid

        src_label = np.zeros((label_h, max_w, 3), dtype=np.uint8)
        cv2.putText(src_label, f"SOURCE FACES ({len(src_faces)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        tgt_label = np.zeros((label_h, max_w, 3), dtype=np.uint8)
        cv2.putText(tgt_label, f"TARGET FACES ({len(tgt_faces)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        combined = np.vstack([src_label, src_grid_padded, tgt_label, tgt_grid_padded])

        ok, buf = cv2.imencode(".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode image."}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="detected_faces.jpg"
        )
    except Exception as e:
        print(f"Error in detect: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap", methods=["POST"])
def swap():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        source_index = int(request.form.get("source_index", 0))
        target_index = request.form.get("target_index", None)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

        if source_index >= len(src_faces):
            source_index = 0
        src_face = src_faces[source_index]

        out_img = tgt_img.copy()
        
        if target_index is not None:
            target_index = int(target_index)
            if target_index < len(tgt_faces):
                out_img = swapper.get(out_img, tgt_faces[target_index], src_face, paste_back=True)
        else:
            for tgt_face in tgt_faces:
                out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)
                break

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image."}), 500
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap.jpg"
        )
    except Exception as e:
        print(f"Error in swap: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap_all", methods=["POST"])
def swap_all():
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        source_index = int(request.form.get("source_index", 0))

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

        if source_index >= len(src_faces):
            source_index = 0
        src_face = src_faces[source_index]

        out_img = tgt_img.copy()
        for tgt_face in tgt_faces:
            out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image."}), 500
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap.jpg"
        )
    except Exception as e:
        print(f"Error in swap_all: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/detect_faces", methods=["POST"])
def detect_faces():
    """
    Detects faces in both source and target images.
    Returns all faces as base64 JPEGs with type identifier (source/target).
    """
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        all_faces = []

        for i, face in enumerate(src_faces):
            crop = extract_face_crop(src_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                all_faces.append({
                    "type": "source",
                    "index": i,
                    "image": b64_str
                })

        for i, face in enumerate(tgt_faces):
            crop = extract_face_crop(tgt_img, face)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
                all_faces.append({
                    "type": "target",
                    "index": i,
                    "image": b64_str
                })

        return jsonify({
            "source_count": len(src_faces),
            "target_count": len(tgt_faces),
            "faces": all_faces
        })
    except Exception as e:
        print(f"Error in detect_faces: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app_flask.route("/swap_selected", methods=["POST"])
def swap_selected():
    """
    Swaps selected faces between source and target images.
    Expects:
      - source: source image file
      - target: target image file
      - swaps: JSON string of swap pairs, e.g. [{"source_index": 0, "target_index": 1}]
    Returns the output image with all specified swaps applied.
    """
    try:
        if not init_models():
            return jsonify({"error": "Failed to load AI models. Please try again."}), 503

        if "source" not in request.files or "target" not in request.files:
            return jsonify({"error": "Missing files. Expect 'source' and 'target'."}), 400

        src_img = read_image_from_request(request.files["source"])
        tgt_img = read_image_from_request(request.files["target"])

        if src_img is None or tgt_img is None:
            return jsonify({"error": "Invalid image data."}), 400

        src_img = resize_image(src_img)
        tgt_img = resize_image(tgt_img)

        swaps_json = request.form.get("swaps", "[]")
        try:
            import json
            swaps = json.loads(swaps_json)
        except:
            return jsonify({"error": "Invalid 'swaps' format. Expected JSON array."}), 400

        if not isinstance(swaps, list) or len(swaps) == 0:
            return jsonify({"error": "No swap pairs provided. Expected array of {source_index, target_index}."}), 400

        src_faces = face_app.get(src_img)
        tgt_faces = face_app.get(tgt_img)

        if len(src_faces) == 0:
            return jsonify({"error": "No face detected in source image."}), 400
        if len(tgt_faces) == 0:
            return jsonify({"error": "No face detected in target image."}), 400

        out_img = tgt_img.copy()

        for swap_pair in swaps:
            source_idx = int(swap_pair.get("source_index", 0))
            target_idx = int(swap_pair.get("target_index", 0))

            if source_idx >= len(src_faces):
                source_idx = 0
            if target_idx >= len(tgt_faces):
                continue

            src_face = src_faces[source_idx]
            tgt_face = tgt_faces[target_idx]
            out_img = swapper.get(out_img, tgt_face, src_face, paste_back=True)

        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return jsonify({"error": "Failed to encode output image."}), 500
        
        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="swap_selected.jpg"
        )
    except Exception as e:
        print(f"Error in swap_selected: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app_flask.run(host="0.0.0.0", port=port, debug=False, threaded=False)
