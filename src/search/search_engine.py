import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# --- THE SOTA FIX: Import the Dedicated Independent Brains! ---
from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection

# --- CONFIGURATION ---
FEATURE_PATH = "../../data/search/gallery_features_clip.npy"
PATHS_PATH = "../../data/search/gallery_paths_clip.npy"

#  SEARCH QUERY: You can search by TEXT or by IMAGE!
QUERY_TEXT = "a person wearing yellow shirt" 
QUERY_IMAGE_PATH = None 

TOP_K = 5 
# ---------------------

def compute_cosine_similarity(query_vector, gallery_vectors):
    """The math that powers Google Search: Compares the angles of the vectors."""
    query_vector = query_vector / np.linalg.norm(query_vector)
    gallery_vectors = gallery_vectors / np.linalg.norm(gallery_vectors, axis=1, keepdims=True)
    return np.dot(gallery_vectors, query_vector)

def plot_results(query_text, query_img_path, result_paths, scores):
    """Displays the top matches in a beautiful grid."""
    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(15, 4))
    
    # 1. Plot the Query (What we are looking for)
    axes[0].axis('off')
    if query_img_path:
        axes[0].imshow(Image.open(query_img_path))
        axes[0].set_title("QUERY IMAGE", color='blue', fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, f'TEXT QUERY:\n"{query_text}"', 
                     fontsize=12, ha='center', va='center', wrap=True)
        axes[0].set_title("QUERY TEXT", color='blue', fontweight='bold')

    # 2. Plot the Results (What the AI found)
    for i in range(TOP_K):
        ax = axes[i + 1]
        ax.imshow(Image.open(result_paths[i]))
        ax.axis('off')
        ax.set_title(f"Match {i+1}\nScore: {scores[i]:.2f}", color='green', fontweight='bold')
        
    plt.tight_layout()
    plt.show()

def run_search():
    print("1. Loading Vector Database...")
    try:
        gallery_features = np.load(FEATURE_PATH)
        gallery_paths = np.load(PATHS_PATH)
    except FileNotFoundError:
        print("❌ ERROR: Database not found. Did you run build_gallery.py?")
        return
        
    print("2. Waking up OpenAI's Dedicated CLIP Brains...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)

    print("3. Processing Search Query...")
    with torch.no_grad():
        if QUERY_IMAGE_PATH:
            print(f"   -> Image Search: {QUERY_IMAGE_PATH}")
            # Load ONLY the Vision Brain
            vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id)
            vision_model.eval()
            
            image = Image.open(QUERY_IMAGE_PATH).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = vision_model(**inputs)
            query_vector = outputs.image_embeds[0].cpu().numpy()
            
        elif QUERY_TEXT:
            print(f"   -> Text Search: '{QUERY_TEXT}'")
            # Load ONLY the Text Brain
            text_model = CLIPTextModelWithProjection.from_pretrained(model_id)
            text_model.eval()
            
            inputs = processor(text=[QUERY_TEXT], return_tensors="pt", padding=True)
            outputs = text_model(**inputs)
            query_vector = outputs.text_embeds[0].cpu().numpy()
            
        else:
            print("❌ ERROR: You must provide either a text or image query!")
            return

    # --- THE FIX IS HERE: Dynamically count the database size! ---
    db_size = len(gallery_features)
    print(f"4. Scanning {db_size:,} CCTV Images...")
    # -------------------------------------------------------------
    
    similarities = compute_cosine_similarity(query_vector, gallery_features)
    
    # Sort the results from highest score to lowest score
    top_indices = np.argsort(similarities)[::-1][:TOP_K]
    
    top_paths = gallery_paths[top_indices]
    top_scores = similarities[top_indices]

    print("✅ SUCCESS! Targets found. Displaying results...")
    plot_results(QUERY_TEXT, QUERY_IMAGE_PATH, top_paths, top_scores)

if __name__ == "__main__":
    run_search()