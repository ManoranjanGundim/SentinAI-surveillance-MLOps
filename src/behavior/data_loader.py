import cv2
import numpy as np
import os

def load_video_frames(video_path, max_frames=30, resize=(224, 224)):
    """Reads a video, resizes frames, and pads/truncates to max_frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224 for EfficientNet
        frame = cv2.resize(frame, resize)
        # Normalize pixel values to [0, 1]
        #frame = frame / 255.0
        frames.append(frame)
        
    cap.release()
    
    # Pad with zeros if video is too short
    while len(frames) < max_frames:
        frames.append(np.zeros((resize[0], resize[1], 3)))
        
    return np.array(frames)

# Test the function
if __name__ == "__main__":
    # Point this to a real video in your dataset
    dummy_path = "../../data/raw/Fight/test_video.mp4" 
    if os.path.exists(dummy_path):
        frames = load_video_frames(dummy_path)
        print(f"Successfully loaded video. Shape: {frames.shape}") # Should be (30, 224, 224, 3)
    else:
        print("Dataset not found yet. Please download RWF-2000 into data/raw/")