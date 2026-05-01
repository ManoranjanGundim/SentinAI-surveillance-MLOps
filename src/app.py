import os
import cv2
import numpy as np
import tensorflow as tf

# --- MLOps Senior Trick: Keras 3 "Monkeypatch" Bypass ---
# Intercepts the Keras 3 load process and deletes the broken Google keyword before it crashes!
original_dense_init = tf.keras.layers.Dense.__init__
def safe_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = safe_dense_init

from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import torch
from collections import deque
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys

# --- EMAIL IMPORTS ---
import smtplib
import ssl
from email.message import EmailMessage
import threading

# Import Module 2
sys.path.append(os.path.join(os.path.dirname(__file__), 'threat'))
try:
    from siamese_model import build_siamese_network
except ImportError:
    print("❌ ERROR: Could not import build_siamese_network.")

# --- CONFIGURATION ---

# This gets the exact path to your 'src' folder dynamically!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Now it builds bulletproof absolute paths for any environment
MOD1_PATH = os.path.join(BASE_DIR, "../models/behaviour_engine_final.h5")
MOD2_WEIGHTS = os.path.join(BASE_DIR, "../models/threat_engine.h5")
REF_IMAGE_PATH = os.path.join(BASE_DIR, "../data/threat/Handguns/Handgun_46.jpeg")
GALLERY_FEATURES = os.path.join(BASE_DIR, "../data/search/gallery_features_clip.npy")
GALLERY_PATHS = os.path.join(BASE_DIR, "../data/search/gallery_paths_clip.npy")

RECORDED_VIDEO_PATH = os.path.join(BASE_DIR, "../data/file_001799.avi")
CURRENT_SOURCE = RECORDED_VIDEO_PATH 
SEQ_LENGTH = 15
TOP_K_SEARCH = 3



# 🚨 EMAIL ALERT SETTINGS 🚨
# Put your email and the 16-letter App Password here!
SENDER_EMAIL = "granjan050804@gmail.com"
EMAIL_PASSWORD = "gpjawqkvaogepoml" 
RECEIVER_EMAIL = "granjan050804@gmail.com" # You can send it to yourself!

app = Flask(__name__)

# --- GLOBAL THREAT STATE ---
suspect_data = {
    "caught": False,
    "suspect_image": "",
    "matches": []
}

print(" LOADING ALL SENTINAI ENGINES FOR WEB...")
behavior_model = tf.keras.models.load_model(MOD1_PATH)

threat_model = build_siamese_network()
threat_model.load_weights(MOD2_WEIGHTS)

# --- THE FIX: Add a safety net so missing images NEVER crash the server! ---
ref_img = cv2.imread(REF_IMAGE_PATH)
if ref_img is None:
    print(f"⚠️ WARNING: Could not find reference image at {REF_IMAGE_PATH}. Using safety blank!")
    ref_img = np.zeros((224, 224, 3), dtype=np.uint8) # Generates a blank black image

ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_img = cv2.resize(ref_img, (224, 224)) / 255.0
ref_img_tensor = np.expand_dims(ref_img, axis=0)

print("-> Loading Vector Database...")
gallery_vectors = np.load(GALLERY_FEATURES)
gallery_paths = np.load(GALLERY_PATHS)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


print("✅ AI LOADED. SERVER STARTING.")

# --- INITIALIZE FULL HUGGINGFACE CLIP AI (VISION + TEXT) ---
# --- INITIALIZE FULL HUGGINGFACE CLIP AI (VISION + TEXT) ---
print("⏳ Loading HuggingFace CLIP Engine...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP Engine Ready!")

def send_alert_thread(threat_type, suspect_frame):
    """Runs in the background to send the email without freezing the web video."""
    print(f"\n📧 DISPATCHING EMAIL ALERT FOR {threat_type.upper()}...")
    # --- FIX: Use absolute path for saving the image too! ---
    evidence_path = os.path.join(BASE_DIR, "static", "email_evidence.jpg")
    cv2.imwrite(evidence_path, suspect_frame)
    
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 URGENT: {threat_type} Detected on Camera 1!"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg.set_content(f"SentinAI has detected a {threat_type}. Please check the attached security camera footage immediately.")
        
        # --- THE FIX: Build a bulletproof absolute path to the newly saved threat image ---
        evidence_path = os.path.join(BASE_DIR, "static", "active_threat.jpg")
        
        with open(evidence_path, "rb") as f:
            img_data = f.read()
            # ... (keep your existing email attachment logic below this)
            
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='suspect_caught.jpg')
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
            
        print("✅ Alert Email Sent Successfully!")
    except Exception as e:
        print(f"❌ Failed to send email. Check credentials. Error: {e}")

def run_clip_search(frame):
    global suspect_data
    
    # 1. Save the new live frame
    static_folder = os.path.join(BASE_DIR, "static")
    os.makedirs(static_folder, exist_ok=True)
    
    suspect_filename = "active_threat.jpg"
    suspect_path = os.path.join(static_folder, suspect_filename)
    cv2.imwrite(suspect_path, frame)
    
    # 2. Convert the image for HuggingFace CLIP
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # 3. THE AI MATH: Extract features and calculate cosine similarity
    matches = []
    try:
        # Load the pre-calculated suspect databases using absolute paths
        gallery_feats = np.load(GALLERY_FEATURES)
        gallery_paths = np.load(GALLERY_PATHS)
        
        # Tell the upgraded model to use its vision engine!
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            live_feat = clip_model.get_image_features(**inputs).numpy()
            
        # Force vectors into percentages (0 to 100)
        live_norm = np.linalg.norm(live_feat)
        gallery_norms = np.linalg.norm(gallery_feats, axis=1)
        similarities = np.dot(gallery_feats, live_feat.T).flatten() / (gallery_norms * live_norm + 1e-8)
        
        top_indices = np.argsort(similarities)[-TOP_K_SEARCH:][::-1]
        
        # PATH FIX: Translate Windows paths to Docker paths
        # --- THE ULTIMATE PATH HUNTER ---
        for i, idx in enumerate(top_indices):
            raw_path = str(gallery_paths[idx])
            
            # Extract just the filename (e.g., "suspect1.jpg")
            filename = os.path.basename(raw_path.replace("\\", "/"))
            
            # Dynamically search the ENTIRE data folder for the missing image
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
                print(f"⚠️ MISSING IMAGE: Could not find {filename} anywhere in the data folder!")
                # Create a blank red placeholder so the web UI never shows a broken icon
                blank = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(blank, "IMG MISSING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(static_match_path, blank)
            
            # Cap the score so it always looks clean
            score = float(similarities[idx]) * 100
            score = min(score, 99.9) 
            
            matches.append({
                "image": f"/static/match_{i}.jpg",
                "score": round(score, 1)
            })
        
    except Exception as e:
        print(f"⚠️ ERROR DURING DATABASE SEARCH: {e}", flush=True) 
    
    # 4. Send everything to the Web Dashboard
    suspect_data["suspect_image"] = f"/static/{suspect_filename}"
    suspect_data["caught"] = True
    suspect_data["matches"] = matches
    print("🚨 SUSPECT DATA SENT TO WEB DASHBOARD!")

def generate_frames():
    global suspect_data, CURRENT_SOURCE
    
    # Track what source the OpenCV camera is currently locked onto
    active_source = CURRENT_SOURCE
    cap = cv2.VideoCapture(active_source)
    buffer = deque(maxlen=SEQ_LENGTH)
    frame_counter = 0
    mod1_status, mod2_status, alert_color = "Scanning...", "Scanning...", (0, 255, 0)

    while True:
        # MAGIC SWITCH: If the user clicked the button, reboot the camera instantly!
        if CURRENT_SOURCE != active_source:
            cap.release()
            active_source = CURRENT_SOURCE
            cap = cv2.VideoCapture(active_source)
            buffer.clear()
            frame_counter = 0
            
        ret, frame = cap.read()
        
        if not ret:
            # If video ends, loop it. If webcam fails, show a black error frame.
            if isinstance(active_source, str): 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                # Docker Windows Error handling: Create a black frame with red text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "WEBCAM NOT FOUND (Docker Limitation)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                ret, buffer_img = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer_img.tobytes() + b'\r\n')
                continue

        # --- EXACT SAME AI LOGIC AS BEFORE ---
        frame_counter += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        buffer.append(resized)

        if len(buffer) == SEQ_LENGTH and frame_counter % 5 == 0:
            is_threat = False
            detected_threat_type = None
            
            # Behavior Check
            seq_array = np.expand_dims(np.array(buffer), axis=0)
            v_pred = behavior_model.predict(seq_array, verbose=0)[0][0]
            if v_pred > 0.60:
                mod1_status = f"VIOLENCE DETECTED ({v_pred*100:.1f}%)"
                is_threat = True
                detected_threat_type = "Violence"
            else:
                mod1_status = f"Normal Activity ({v_pred*100:.1f}%)"

            # Threat Check
            mod2_tensor = np.expand_dims(resized / 255.0, axis=0)
            w_pred = threat_model.predict([ref_img_tensor, mod2_tensor], verbose=0)[0][0]
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

# --- FLASK ROUTES ---
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
        # Replace this URL with the exact IP address your phone app gives you!
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
        
        # --- THE FIX: Create absolute path and auto-generate the folder! ---
        static_folder = os.path.join(BASE_DIR, "static")
        os.makedirs(static_folder, exist_ok=True) # Creates the folder if it's missing
        
        save_path = os.path.join(static_folder, f"uploaded_video{ext}")
        file.save(save_path)
        # -------------------------------------------------------------------
        
        CURRENT_SOURCE = save_path
        suspect_data["caught"] = False 
        
        print(f"📥 New video uploaded and active: {save_path}")
        return jsonify({"status": "success", "source": CURRENT_SOURCE})
    
@app.route('/api/search_text', methods=['POST'])
def search_text():
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"status": "error", "message": "No prompt provided"})

    try:
        # 1. Load the suspect database
        gallery_feats = np.load(GALLERY_FEATURES)
        gallery_paths = np.load(GALLERY_PATHS)

        # 2. Convert the text prompt into an AI vector
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            # Get text features instead of image features!
            text_features = clip_model.get_text_features(**inputs).numpy()

        # 3. Bulletproof Match Math
        text_norm = np.linalg.norm(text_features)
        gallery_norms = np.linalg.norm(gallery_feats, axis=1)
        similarities = np.dot(gallery_feats, text_features.T).flatten() / (gallery_norms * text_norm + 1e-8)

        top_indices = np.argsort(similarities)[-TOP_K_SEARCH:][::-1]

        # 4. Grab the images using the ULTIMATE PATH HUNTER!
        results = []
        static_folder = os.path.join(BASE_DIR, "static")
        os.makedirs(static_folder, exist_ok=True)

        for i, idx in enumerate(top_indices):
            raw_path = str(gallery_paths[idx])
            filename = os.path.basename(raw_path.replace("\\", "/"))
            
            # Dynamically search the ENTIRE data folder for the missing image
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
                print(f"⚠️ TEXT SEARCH MISSING IMAGE: Could not find {filename}")
                blank = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.putText(blank, "IMG MISSING", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(static_match_path, blank)
                
            # Scale the CLIP score up for the UI (treat 0.35 raw similarity as 100%)
            raw_score = float(similarities[idx])
            scaled_score = (raw_score / 0.35) * 100
            score = min(scaled_score, 99.9)
            results.append({
                "image": f"/static/text_match_{i}.jpg",
                "score": round(score, 1)
            })

        print(f"🔎 Text Search Results for '{query}': {results}")
        return jsonify({"status": "success", "matches": results})
        
    except Exception as e:
        print(f"⚠️ TEXT SEARCH ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("\n🚀 LAUNCHING WEB SERVER! Open your browser to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)