import os
import cv2
import numpy as np
import tensorflow as tf

# Import your custom Siamese architecture for Mod 2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/threat'))
from siamese_model import build_siamese_network

# --- CONFIGURATION ---
MOD1_PATH = "models/sentinai_final_brain.h5"
MOD2_WEIGHTS = "models/threat_engine.h5"
REFERENCE_IMAGE = "data/threat/Handguns/image_001.jpg" # Update this to your gun image!

# Test Data Directories
MOD1_TEST_DIR = "data/test_behavior"
MOD2_TEST_DIR = "data/test_threat"
# ---------------------

def evaluate_module_1():
    print("\n" + "="*50)
    print("🧠 EVALUATING MODULE 1: BEHAVIOR ENGINE")
    print("="*50)
    
    if not os.path.exists(MOD1_TEST_DIR):
        print(f"❌ ERROR: Test directory {MOD1_TEST_DIR} not found.")
        return

    print("Loading 92% Behavior Engine...")
    try:
        model = tf.keras.models.load_model(MOD1_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    correct_predictions = 0
    total_videos = 0

    # Classes: Fight = 1, Safe = 0
    classes = {'Fight': 1, 'Safe': 0}
    
    for class_name, label in classes.items():
        folder_path = os.path.join(MOD1_TEST_DIR, class_name)
        if not os.path.exists(folder_path): continue
            
        for video_file in os.listdir(folder_path):
            if not video_file.endswith(('.mp4', '.avi')): continue
                
            video_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            frames = []
            while len(frames) < 15:
                ret, frame = cap.read()
                if not ret: break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_frame, (224, 224))
                frames.append(resized)
            cap.release()

            # Pad if video is too short
            while len(frames) < 15: frames.append(np.zeros((224, 224, 3)))

            input_sequence = np.expand_dims(np.array(frames), axis=0)
            
            # Predict
            pred_score = model.predict(input_sequence, verbose=0)[0][0]
            pred_label = 1 if pred_score > 0.50 else 0
            
            total_videos += 1
            if pred_label == label:
                correct_predictions += 1
                print(f"✅ {video_file} -> Correct")
            else:
                print(f"❌ {video_file} -> Incorrect (Guessed: {pred_label}, Actual: {label})")

    if total_videos > 0:
        accuracy = (correct_predictions / total_videos) * 100
        print(f"\n📊 MODULE 1 FINAL ACCURACY: {accuracy:.2f}% ({correct_predictions}/{total_videos})")
    else:
        print("No videos found to test!")


def evaluate_module_2():
    print("\n" + "="*50)
    print("🔫 EVALUATING MODULE 2: THREAT ENGINE")
    print("="*50)
    
    if not os.path.exists(MOD2_TEST_DIR):
        print(f"❌ ERROR: Test directory {MOD2_TEST_DIR} not found.")
        return

    print("Loading Siamese Network...")
    model = build_siamese_network()
    model.load_weights(MOD2_WEIGHTS)
    
    # Load Reference Image
    ref_img = cv2.imread(REFERENCE_IMAGE)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.resize(ref_img, (224, 224)) / 255.0
    ref_tensor = np.expand_dims(ref_img, axis=0)

    correct_predictions = 0
    total_images = 0

    # Classes: Threat = 1, Safe = 0
    classes = {'Threat': 1, 'Safe': 0}

    for class_name, label in classes.items():
        folder_path = os.path.join(MOD2_TEST_DIR, class_name)
        if not os.path.exists(folder_path): continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            test_img = cv2.imread(img_path)
            if test_img is None: continue
                
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            test_img = cv2.resize(test_img, (224, 224)) / 255.0
            test_tensor = np.expand_dims(test_img, axis=0)

            # Predict
            pred_score = model.predict([ref_tensor, test_tensor], verbose=0)[0][0]
            pred_label = 1 if pred_score > 0.65 else 0
            
            total_images += 1
            if pred_label == label:
                correct_predictions += 1
                # print(f"✅ {img_file} -> Correct")
            else:
                print(f"❌ {img_file} -> Incorrect (Score: {pred_score:.2f}, Actual: {label})")

    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"\n📊 MODULE 2 FINAL ACCURACY: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    else:
        print("No images found to test!")

if __name__ == "__main__":
    evaluate_module_1()
    evaluate_module_2()