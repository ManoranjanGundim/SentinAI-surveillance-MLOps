import os
import numpy as np
import time
from PIL import Image
import torch

# THE SOTA FIX: Import the Dedicated Vision Extractor!
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

# --- CONFIGURATION ---
GALLERY_DIR = "../../data/search/bounding_box_test"
FEATURE_SAVE_PATH = "../../data/search/gallery_features_clip.npy"
PATHS_SAVE_PATH = "../../data/search/gallery_paths_clip.npy"
MAX_IMAGES = 19000 
# ---------------------

def build_database():
    print("1. Initializing OpenAI's Dedicated CLIP Vision Brain...")
    model_id = "openai/clip-vit-base-patch32"
    
    # This specific model GUARANTEES a clean 512-D vector output
    model = CLIPVisionModelWithProjection.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    model.eval() # Lock the brain into 'read-only' mode
    
    print(f"\n2. Scanning CCTV Gallery Directory: {GALLERY_DIR}")
    if not os.path.exists(GALLERY_DIR):
        print("❌ ERROR: Gallery folder not found! Check your paths.")
        return

    image_filenames = [f for f in os.listdir(GALLERY_DIR) if f.endswith('.jpg')]
    image_filenames = image_filenames[:MAX_IMAGES] 
    
    print(f"3. Found {len(image_filenames)} images. Starting Feature Extraction...")
    
    features_list = []
    paths_list = []
    
    start_time = time.time()
    
    for i, filename in enumerate(image_filenames):
        img_path = os.path.join(GALLERY_DIR, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                # Pass the image through the vision-only brain
                outputs = model(**inputs)
            
            # Directly grab the perfect 512-dimensional vector!
            vector = outputs.image_embeds[0].cpu().numpy()
            
            features_list.append(vector)
            paths_list.append(img_path)
            
            if (i + 1) % 100 == 0:
                print(f"   -> Processed {i + 1} / {len(image_filenames)} images...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n4. Saving CLIP Vector Database to hard drive...")
    np.save(FEATURE_SAVE_PATH, np.array(features_list))
    np.save(PATHS_SAVE_PATH, np.array(paths_list))
    
    elapsed = time.time() - start_time
    print(f"✅ SUCCESS! Extracted CLIP features for {len(features_list)} people in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    build_database()