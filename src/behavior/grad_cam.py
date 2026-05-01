import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "../../models/behaviour_engine_final.h5"
SEQUENCE_LENGTH = 15
IMG_SIZE = 224

print("1. Waking up Explainable AI Engine...")
model = tf.keras.models.load_model(MODEL_PATH)

print("2. Extracting SentinAI's Visual Cortex...")
# We must dig inside the TimeDistributed layer to find the EfficientNet!
cnn_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.TimeDistributed):
        cnn_layer = layer.layer
        break

# EfficientNetB0's final visual layer is called 'top_activation'
grad_model = tf.keras.models.Model(
    [cnn_layer.inputs], 
    [cnn_layer.get_layer('top_activation').output, cnn_layer.output]
)

def get_img_array(frame, size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img

def make_spatial_heatmap(img_array):
    """Generates a heatmap of where the AI's visual cortex is activating."""
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Run the frame through the CNN to get the visual feature maps
    conv_outputs, _ = grad_model(img_tensor)
    
    # Compress the 1280 feature channels into a single 2D heatmap
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = tf.squeeze(heatmap)

    # Normalize the glowing colors to 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_frame, alpha=0.5):
    """Glues the glowing heatmap over the original video frame."""
    heatmap = np.uint8(255 * heatmap)
    
    # Use the 'JET' colormap (Blue = Cold/Ignore, Red = Hot/Focus)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Resize the heatmap to match the video frame
    jet_heatmap = cv2.resize(jet_heatmap, (original_frame.shape[1], original_frame.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    
    original_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
    
    # Superimpose the colors
    superimposed_img = jet_heatmap * alpha + original_bgr
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def run_explainable_video():
    video_path = "C:/Users/manor/OneDrive/Desktop/SDP/SentinAI/data/raw/Fight/file_000021.avi" 
    cap = cv2.VideoCapture(video_path)
    
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    print("3. Analyzing Video. Press 'q' to quit, 'p' to pause.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video Finished.")
            break
            
        processed_frame = get_img_array(frame, IMG_SIZE)
        frame_buffer.append(processed_frame)
        
        display_frame = frame.copy()
        
        if len(frame_buffer) == SEQUENCE_LENGTH:
            input_sequence = np.expand_dims(np.array(frame_buffer), axis=0)
            prediction = model.predict(input_sequence, verbose=0)[0][0]
            
            # IF VIOLENCE IS DETECTED, TURN ON THE X-RAY!
            if prediction > 0.50: 
                # Grab the exact middle frame of the fight (Frame 7)
                target_frame = frame_buffer[7] 
                
                heatmap = make_spatial_heatmap(target_frame)
                
                # Turn the target frame back into normal BGR for OpenCV
                base_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2BGR)
                display_frame = overlay_heatmap(heatmap, target_frame)
                
                # Make the frame bigger so you can see the glowing fists
                display_frame = cv2.resize(display_frame, (600, 600))
                
                cv2.putText(display_frame, f"VIOLENCE MATCH: {prediction*100:.1f}%", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                display_frame = cv2.resize(display_frame, (600, 600))
                cv2.putText(display_frame, f"NORMAL: {prediction*100:.1f}%", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('SentinAI - Explainable AI (X-Ray)', display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1) 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_explainable_video()