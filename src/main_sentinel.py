import cv2
import numpy as np
import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# --- NEW IMPORTS FOR ALERTS ---
import smtplib
import ssl
from email.message import EmailMessage
import winsound
import threading

# Import Module 2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'threat'))
try:
    from siamese_model import build_siamese_network
except ImportError:
    print("❌ ERROR: Could not import build_siamese_network.")

# --- 1. CONFIGURATION ---
MOD1_PATH = "../models/behaviour_engine_final.h5"
MOD2_WEIGHTS = "../models/threat_engine.h5"
REF_IMAGE_PATH = "../data/threat/Handguns/Handgun_46.jpeg"

GALLERY_FEATURES = "../data/search/gallery_features_clip.npy"
GALLERY_PATHS = "../data/search/gallery_paths_clip.npy"
VIDEO_SOURCE = "C:/AI_Test/file_001799.avi" # Use 0 for live webcam

SEQ_LENGTH = 15
TOP_K_SEARCH = 3

#  EMAIL ALERT SETTINGS 
# Put your email and the 16-letter App Password here!
SENDER_EMAIL = "granjan050804@gmail.com"
EMAIL_PASSWORD = "gpjawqkvaogepoml" 
RECEIVER_EMAIL = "granjan050804@gmail.com" # You can send it to yourself!

print(" INITIALIZING SENTINAI MASTER CONTROL PROGRAM...")

print("-> Loading Mod 1: Behavior Engine...")
behavior_model = tf.keras.models.load_model(MOD1_PATH)

print("-> Loading Mod 2: Siamese Threat Engine...")
threat_model = build_siamese_network()
threat_model.load_weights(MOD2_WEIGHTS)

ref_img = cv2.imread(REF_IMAGE_PATH)
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_img = cv2.resize(ref_img, (224, 224)) / 255.0
ref_img_tensor = np.expand_dims(ref_img, axis=0)

print("-> Loading Mod 3: Vector Database & CLIP...")
gallery_vectors = np.load(GALLERY_FEATURES)
gallery_paths = np.load(GALLERY_PATHS)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
clip_vision.eval()

print("-> Warming up Neural Networks...")
dummy_seq = np.zeros((1, SEQ_LENGTH, 224, 224, 3), dtype=np.float32)
dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = behavior_model.predict(dummy_seq, verbose=0)
_ = threat_model.predict([ref_img_tensor, dummy_img], verbose=0)
print(" ALL SYSTEMS ONLINE.")


def send_alert_thread(threat_type, suspect_frame):
    """Runs in the background so the video doesn't freeze!"""
    print("\n INITIATING ALARM AND DISPATCHING EMAIL ALERT...")
    
    # 1. Sound the Windows Alarm! (Frequency 2500Hz, Duration 1000ms)
    winsound.Beep(2500, 1000) 
    
    # 2. Save the evidence photo
    evidence_path = "suspect_evidence.jpg"
    cv2.imwrite(evidence_path, suspect_frame)
    
    # 3. Send the Email
    try:
        msg = EmailMessage()
        msg['Subject'] = f"🚨 URGENT: {threat_type} Detected on Camera 1!"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg.set_content(f"SentinAI has detected a {threat_type}. Please check the attached security camera footage immediately.")
        
        with open(evidence_path, 'rb') as f:
            img_data = f.read()
            
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename='suspect.jpg')
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
            
        print("✅ Alert Email Sent Successfully to your phone!")
    except Exception as e:
        print(f" Failed to send email. Check your password. Error: {e}")


def run_suspect_search(suspect_frame):
    """MODULE 3: Triggers automatically when a threat is detected!"""
    print("\n INITIATING DATABASE SUSPECT SEARCH...")
    image = Image.fromarray(cv2.cvtColor(suspect_frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = clip_vision(**inputs)
        query_vector = outputs.image_embeds[0].cpu().numpy()
        
    query_vector = query_vector / np.linalg.norm(query_vector)
    gal_vecs = gallery_vectors / np.linalg.norm(gallery_vectors, axis=1, keepdims=True)
    similarities = np.dot(gal_vecs, query_vector)
    
    top_indices = np.argsort(similarities)[::-1][:TOP_K_SEARCH]
    
    fig, axes = plt.subplots(1, TOP_K_SEARCH + 1, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("CAUGHT SUSPECT", color='red', fontweight='bold')
    axes[0].axis('off')
    
    for i, idx in enumerate(top_indices):
        ax = axes[i + 1]
        raw_path = str(gallery_paths[idx])
        fixed_path = raw_path.replace("../../data", "../data") 
        try:
            ax.imshow(Image.open(fixed_path))
        except:
            pass
        ax.set_title(f"Match {i+1} ({similarities[idx]:.2f})", color='green')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show() 

def start_system():
    cap = cv2.VideoCapture(VIDEO_SOURCE) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    buffer = deque(maxlen=SEQ_LENGTH)
    
    mod1_status = "Mod 1: Scanning..."
    mod2_status = "Mod 2: Scanning..."
    alert_color = (0, 255, 0)
    suspect_caught = False
    frame_counter = 0

    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        for _ in range(SEQ_LENGTH):
            buffer.append(resized_frame)
            
    print("\n SYSTEM ACTIVE: Playing video at normal speed...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame_counter += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        buffer.append(resized_frame)

        mod2_img = resized_frame / 255.0
        mod2_tensor = np.expand_dims(mod2_img, axis=0)

        # Scan every 5 frames
        if len(buffer) == SEQ_LENGTH and frame_counter % 5 == 0:
            detected_threat_type = None
            
            seq_array = np.expand_dims(np.array(buffer), axis=0)
            v_pred = behavior_model.predict(seq_array, verbose=0)[0][0]
            if v_pred > 0.60:
                mod1_status = f"VIOLENCE DETECTED! ({v_pred*100:.1f}%)"
                detected_threat_type = "Violence"
            else:
                mod1_status = f"Behavior Normal ({v_pred*100:.1f}%)"

            w_pred = threat_model.predict([ref_img_tensor, mod2_tensor], verbose=0)[0][0]
            if w_pred > 0.65:
                mod2_status = f"WEAPON DETECTED! ({w_pred*100:.1f}%)"
                detected_threat_type = "Weapon"
            else:
                mod2_status = f"No Weapons ({w_pred*100:.1f}%)"

            # --- THE TRAPDOOR ---
            if detected_threat_type and not suspect_caught:
                alert_color = (0, 0, 255) 
                suspect_caught = True 
                
                # 1. Fire off the Email and Alarm in the background!
                threading.Thread(target=send_alert_thread, args=(detected_threat_type, frame.copy())).start()
                
                # 2. Pop up the Database Search visually
                cv2.putText(frame, "SYSTEM LOCKED - THREAT DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                run_suspect_search(frame) 
                
            elif not detected_threat_type:
                alert_color = (0, 255, 0) 

        cv2.putText(frame, mod1_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        cv2.putText(frame, mod2_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
        cv2.imshow("SentinAI - Unified Command Center", frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_system()