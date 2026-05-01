import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "../../models/behaviour_engine_final.h5" 
SEQUENCE_LENGTH = 15 
IMG_SIZE = 224

print("1. Waking up the GRU Behavior Engine...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Brain loaded successfully!")

def start_video_test():
    
    video_source = 0 
    
    cap = cv2.VideoCapture(video_source) 
    
    # Get the FPS of the video to play it at normal speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    print("2. Initializing live camera feed. Press 'q' to quit.")
    
    # --- UI Variables ---
    frame_counter = 0
    predict_interval = 5  # The AI will only do math every 5 frames!
    current_text = "Buffering AI..."
    current_color = (255, 255, 0) # Cyan while waiting
    

    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        for _ in range(SEQUENCE_LENGTH):
            frame_buffer.append(resized_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        
        frame_buffer.append(resized_frame)
        frame_counter += 1
        
        # 2. Predict ONLY when it's the 5th frame
        if len(frame_buffer) == SEQUENCE_LENGTH:
            if frame_counter % predict_interval == 0:
                frames_array = np.array(frame_buffer)
                input_sequence = np.expand_dims(frames_array, axis=0)
                
                # The AI does its math here (might cause a tiny micro-stutter, but highly reliable)
                prediction = model.predict(input_sequence, verbose=0)[0][0]
                
                # Update the persistent UI text
                if prediction > 0.50: 
                    current_text = f"VIOLENCE DETECTED ({prediction*100:.1f}%)"
                    current_color = (0, 0, 255) # Red
                else:
                    current_text = f"Normal Activity ({prediction*100:.1f}%)"
                    current_color = (0, 255, 0) # Green
                
        # 3. Always draw the LAST known text so the video doesn't look blank
        cv2.putText(frame, current_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 3)
        cv2.imshow('SentinAI - Module 1: Behavior Engine', frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_video_test()