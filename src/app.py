import os
import cv2
from datetime import datetime
import numpy as np
import tensorflow as tf
import sys
import smtplib
import ssl
from email.message import EmailMessage
import threading
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import torch
from collections import deque
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

original_dense_init = tf.keras.layers.Dense.__init__
def safe_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = safe_dense_init

sys.path.append(os.path.join(os.path.dirname(__file__), 'threat'))
try:
    from siamese_model import build_siamese_network
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOD1_PATH = os.path.join(BASE_DIR, "../models/behaviour_engine_final.h5")
MOD2_WEIGHTS = os.path.join(BASE_DIR, "../models/threat_engine.h5")
REF_IMAGE_PATH = os.path.join(BASE_DIR, "../data/threat/Handguns/Handgun_46.jpeg")
GALLERY_FEATURES = os.path.join(BASE_DIR, "../data/search/gallery_features_clip.npy")
GALLERY_PATHS = os.path.join(BASE_DIR, "../data/search/gallery_paths_clip.npy")

RECORDED_VIDEO_PATH = os.path.join(BASE_DIR, "../data/file_001799.avi")
CURRENT_SOURCE = RECORDED_VIDEO_PATH 
SEQ_LENGTH = 15
TOP_K_SEARCH = 3

SENDER_EMAIL = "granjan050804@gmail.com"
EMAIL_PASSWORD = "gpjawqkvaogepoml" 
RECEIVER_EMAIL = "granjan050804@gmail.com" 

app = Flask(__name__)

suspect_data = {
    "caught": False,
    "suspect_image": "",
    "matches": []
}

behavior_model = tf.keras.models.load_model(MOD1_PATH)
threat_model = build_siamese_network()
threat_model.load_weights(MOD2_WEIGHTS)

ref_img = cv2.imread(REF_IMAGE_PATH)
if ref_img is None:
    ref_img = np.zeros((224, 224, 3), dtype=np.uint8)

ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_img = cv2.resize(ref_img, (224, 224)) / 255.0
ref_img_tensor = np.expand_dims(ref_img, axis=0)

gallery_vectors = np.load(GALLERY_FEATURES)
gallery_paths = np.load(GALLERY_PATHS)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs("flagged_data", exist_ok=True)

def flag_low_confidence_data(image_path, search_query, score):
    if 40.0 <= score <= 65.0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        flagged_img_path = f"flagged_data/confused_{timestamp}.jpg"
        img = cv2.imread(image_path)
        if img is not None:
            cv2.imwrite(flagged_img_path, img)
            with open("flagged_data/drift_log.txt", "a") as log:
                log.write(f"[{timestamp}] Query: '{search_query}' | Score: {score:.1f}% | Saved: {flagged_img_path}\n")

def send_alert_thread(threat_type, suspect_frame):
    evidence_path = os.path.join(BASE_DIR, "static", "email_evidence.jpg")
    cv2.imwrite(evidence_path, suspect_frame)
    try:
        msg = EmailMessage()
        msg['Subject'] = f"URGENT: {threat_type} Detected on Camera 1!"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg.set_content(f"SentinAI has detected a {threat_type}. Please check the attached security camera footage immediately.")
        
        evidence_path = os.path.join(BASE_DIR, "static", "active_threat.jpg")
        
        with open(evidence_path, "rb") as f:
            img_data = f.read()
            
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='suspect_caught.jpg')
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception:
        pass

def run_clip_search(frame):
    global suspect_data
    static_folder = os.path.join(BASE_DIR, "static")
    os.makedirs(static_folder, exist_ok=True)
    suspect_filename = "active_threat.jpg"
    suspect_path = os.path.join(static_folder, suspect_filename)
    cv2.imwrite(suspect_path, frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    matches = []
    try:
        gallery_feats = np.load(GALLERY_FEATURES)
        gallery_paths = np.load(GALLERY_PATHS)
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            live_feat = clip_model.get_image_features(**inputs).numpy()
        live_norm = np.linalg.norm(live_feat)
        gallery_norms = np.linalg.norm(gallery_feats, axis=1)
        similarities = np.dot(gallery_feats, live_feat.T).flatten() / (gallery_norms * live_norm + 1e-8)
        top_indices = np.argsort(similarities)[-TOP_K_SEARCH:][::-1]
        for i, idx in enumerate(top_indices):
            raw_path = str(gallery_paths[idx])
            filename = os.path.basename(raw_path.replace("\\", "/"))
            data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
            found_path = None
            for root, dirs, files in os.walk(data_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    break
            static_match_path = os.path.join(static_folder, f"match_{i}.jpg")
            if found_path is not None:
                match_img = cv2.imread(found_path)
                cv2.imwrite(static_match_path, match_img)
            else:
                blank = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(blank, "IMG MISSING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(static_match_path, blank)
            score = float(similarities[idx]) * 100
            score = min(score, 99.9) 
            
            flag_low_confidence_data(static_match_path, "Image Reference", score)

            matches.append({
                "image": f"/static/match_{i}.jpg",
                "score": round(score, 1)
            })
    except Exception:
        pass 
    suspect_data["suspect_image"] = f"/static/{suspect_filename}"
    suspect_data["caught"] = True
    suspect_data["matches"] = matches

def generate_frames():
    global suspect_data, CURRENT_SOURCE
    active_source = CURRENT_SOURCE
    cap = cv2.VideoCapture(active_source)
    buffer = deque(maxlen=SEQ_LENGTH)
    frame_counter = 0
    mod1_status, mod2_status, alert_color = "Scanning...", "Scanning...", (0, 255, 0)
    while True:
        if CURRENT_SOURCE != active_source:
            cap.release()
            active_source = CURRENT_SOURCE
            cap = cv2.VideoCapture(active_source)
            buffer.clear()
            frame_counter = 0
        ret, frame = cap.read()
        if not ret:
            if isinstance(active_source, str): 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "WEBCAM NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                ret, buffer_img = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')
                continue
        frame_counter += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        buffer.append(resized)
        if len(buffer) == SEQ_LENGTH and frame_counter % 5 == 0:
            is_threat = False
            detected_threat_type = None
            seq_array = np.expand_dims(np.array(buffer), axis=0)
            v_pred = behavior_model.predict(seq_array, verbose=0)[0][0]
            
            if 0.40 <= v_pred <= 0.65:
                temp_path = os.path.join(BASE_DIR, "static", "temp_behavior.jpg")
                cv2.imwrite(temp_path, frame)
                flag_low_confidence_data(temp_path, "Behavior Detection", v_pred * 100)

            if v_pred > 0.60:
                mod1_status = f"VIOLENCE DETECTED ({v_pred*100:.1f}%)"
                is_threat = True
                detected_threat_type = "Violence"
            else:
                mod1_status = f"Normal Activity ({v_pred*100:.1f}%)"
            mod2_tensor = np.expand_dims(resized / 255.0, axis=0)
            w_pred = threat_model.predict([ref_img_tensor, mod2_tensor], verbose=0)[0][0]

            if 0.40 <= w_pred <= 0.65:
                temp_path = os.path.join(BASE_DIR, "static", "temp_weapon.jpg")
                cv2.imwrite(temp_path, frame)
                flag_low_confidence_data(temp_path, "Weapon Detection", w_pred * 100)

            if w_pred > 0.65:
                mod2_status = f"WEAPON DETECTED ({w_pred*100:.1f}%)"
                is_threat = True
                detected_threat_type = "Weapon"
            else:
                mod2_status = f"No Weapons ({w_pred*100:.1f}%)"
            if is_threat and not suspect_data["caught"]:
                alert_color = (0, 0, 255)
                run_clip_search(frame.copy())
                threading.Thread(target=send_alert_thread, args=(detected_threat_type, frame.copy())).start()
            elif not is_threat and not suspect_data["caught"]:
                alert_color = (0, 255, 0)
        cv2.putText(frame, mod1_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        cv2.putText(frame, mod2_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        ret, buffer_img = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')

@app.route('/')
def index():
    global suspect_data
    suspect_data["caught"] = False 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/threat_status')
def threat_status():
    return jsonify(suspect_data)

@app.route('/data_files/<path:filename>')
def serve_data(filename):
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    return send_from_directory(data_dir, filename)

@app.route('/api/set_source/<source_type>')
def set_source(source_type):
    global CURRENT_SOURCE, suspect_data
    suspect_data["caught"] = False 
    if source_type == 'webcam':
        CURRENT_SOURCE = "http://10.192.215.150:8080/video" 
    else:
        CURRENT_SOURCE = RECORDED_VIDEO_PATH
    return jsonify({"status": "success", "source": str(CURRENT_SOURCE)})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global CURRENT_SOURCE, suspect_data
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
    if file:
        ext = os.path.splitext(file.filename)[1]
        static_folder = os.path.join(BASE_DIR, "static")
        os.makedirs(static_folder, exist_ok=True)
        save_path = os.path.join(static_folder, f"uploaded_video{ext}")
        file.save(save_path)
        CURRENT_SOURCE = save_path
        suspect_data["caught"] = False 
        return jsonify({"status": "success", "source": CURRENT_SOURCE})
    
@app.route('/api/search_text', methods=['POST'])
def search_text():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"status": "error", "message": "No prompt provided"})
    try:
        gallery_feats = np.load(GALLERY_FEATURES)
        gallery_paths = np.load(GALLERY_PATHS)
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs).numpy()
        text_norm = np.linalg.norm(text_features)
        gallery_norms = np.linalg.norm(gallery_feats, axis=1)
        similarities = np.dot(gallery_feats, text_features.T).flatten() / (gallery_norms * text_norm + 1e-8)
        top_indices = np.argsort(similarities)[-TOP_K_SEARCH:][::-1]
        results = []
        static_folder = os.path.join(BASE_DIR, "static")
        os.makedirs(static_folder, exist_ok=True)
        for i, idx in enumerate(top_indices):
            raw_path = str(gallery_paths[idx])
            filename = os.path.basename(raw_path.replace("\\", "/"))
            data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
            found_path = None
            for root, dirs, files in os.walk(data_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    break
            static_match_path = os.path.join(static_folder, f"text_match_{i}.jpg")
            if found_path is not None:
                match_img = cv2.imread(found_path)
                cv2.imwrite(static_match_path, match_img)
            else:
                blank = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(blank, "IMG MISSING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(static_match_path, blank)
            raw_score = float(similarities[idx])
            scaled_score = (raw_score / 0.35) * 100
            score = min(scaled_score, 99.9)
            
            flag_low_confidence_data(static_match_path, query, score)

            results.append({
                "image": f"/static/text_match_{i}.jpg",
                "score": round(score, 1)
            })
        return jsonify({"status": "success", "matches": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)