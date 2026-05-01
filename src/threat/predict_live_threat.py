import cv2
import numpy as np
import tensorflow as tf
import os
from siamese_model import build_siamese_network

# --- CONFIGURATION ---
MODEL_WEIGHTS_PATH = "../../models/threat_engine.h5"

# Pick ONE image from your dataset to act as the "Target Threat"
# Make sure to update this path to an actual image in your Handguns folder!
REFERENCE_IMAGE_PATH = "../../data/threat/Test_AI/terrorists2.jpg" 
# ---------------------

def preprocess_image(img, img_size=(224, 224)):
    """Formats an image exactly how MobileNetV2 expects it."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0) # Add batch dimension

def main():
    print("1. Waking up Twin Brains...")
    model = build_siamese_network()
    
    print("2. Loading SOTA 98.8% Weights...")
    try:
        model.load_weights(MODEL_WEIGHTS_PATH)
    except Exception as e:
        print(f"❌ Error loading weights. Did you run train_threat.py successfully? Error: {e}")
        return

    print("3. Loading Reference Threat Image...")
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"❌ ERROR: Could not find reference image at {REFERENCE_IMAGE_PATH}")
        print("Please update REFERENCE_IMAGE_PATH to point to a real image in your dataset.")
        return
        
    ref_img_raw = cv2.imread(REFERENCE_IMAGE_PATH)
    ref_img_processed = preprocess_image(ref_img_raw)
    
    print("4. Initializing Live Camera Feed...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        return

    print("🟢 SYSTEM ACTIVE: Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess the live webcam frame
        live_frame_processed = preprocess_image(frame)
        
        # Ask the Siamese Network to compare them!
        prediction = model.predict([ref_img_processed, live_frame_processed], verbose=0)
        score = prediction[0][0] # The math score between 0 (Mismatch) and 1 (Match)
        
        # --- UI Overlay ---
        # If score is closer to 1, it's a match!
        if score > 0.65: 
            label = f"THREAT MATCH! ({score*100:.1f}%)"
            color = (0, 0, 255) # Red for danger
        else:
            label = f"SAFE - No Match ({score*100:.1f}%)"
            color = (0, 255, 0) # Green for safe
            
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Show the reference image in the top right corner so you know what it's looking for
        ref_display = cv2.resize(ref_img_raw, (150, 150))
        frame[10:160, -160:-10] = ref_display
        
        cv2.imshow("SentinAI - Module 2: Siamese Threat Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()