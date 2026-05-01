import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from siamese_loader import SiameseDataGenerator
from siamese_model import build_siamese_network

# --- CONFIGURATION ---
DATA_DIR = "../../data/threat"
MODEL_SAVE_PATH = "../../models/threat_engine.h5"
BATCH_SIZE = 16
EPOCHS = 30

if __name__ == "__main__":
    print("Initializing Siamese Training Pipeline...")
    
    # 1. Load Data
    train_gen = SiameseDataGenerator(DATA_DIR, batch_size=BATCH_SIZE)
    
    # 2. Build Model
    model = build_siamese_network()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    # 3. Setup Callbacks
    os.makedirs("../../models", exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    early_stop = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True, verbose=1)
    
    # 4. Train
    print("\n--- Starting Siamese Twin Brain Training ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )
    
    print(f"\nTraining Complete! SOTA Threat Model saved to {MODEL_SAVE_PATH}")