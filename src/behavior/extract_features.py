import os
import numpy as np
import tensorflow as tf
from data_loader import load_video_frames

# --- CONFIGURATION ---
DATA_DIR = "../../data/raw"
FEATURE_DIR = "../../data/features"
SEQUENCE_LENGTH = 30
IMG_SIZE = 224

def build_extractor():
    print("Loading EfficientNet Extractor...")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', pooling='avg'
    )
    return base_model

def extract_and_save():
    extractor = build_extractor()
    classes = {'NonFight': 0, 'Fight': 1}
    
    # Create feature directories
    for class_name in classes.keys():
        os.makedirs(os.path.join(FEATURE_DIR, class_name), exist_ok=True)
        
    for class_name in classes.keys():
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):
            continue
            
        video_files = os.listdir(class_path)
        print(f"Extracting features for {len(video_files)} videos in {class_name}...")
        
        for i, video_file in enumerate(video_files):
            # Check if we already extracted it (resume capability!)
            feature_path = os.path.join(FEATURE_DIR, class_name, video_file + ".npy")
            if os.path.exists(feature_path):
                continue
                
            video_path = os.path.join(class_path, video_file)
            try:
                # Load 30 frames
                frames = load_video_frames(video_path, max_frames=SEQUENCE_LENGTH, resize=(IMG_SIZE, IMG_SIZE))
                
                # Predict (Extract features) -> Output shape will be (30, 1280)
                features = extractor.predict(frames, verbose=0)
                
                # Save to disk as a tiny numpy file
                np.save(feature_path, features)
                
                if i % 10 == 0:
                    print(f"Processed {i}/{len(video_files)} {class_name} videos...")
            except Exception as e:
                print(f"Error on {video_file}: {e}")

if __name__ == "__main__":
    extract_and_save()
    print("EXTRACTION COMPLETE! You are ready for Lightning-Fast Training.")