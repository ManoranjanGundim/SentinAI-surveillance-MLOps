import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Dropout, Dense, BatchNormalization, GaussianNoise, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# --- CONFIGURATION ---
FEATURE_DIR = "../../data/features"
MODEL_SAVE_PATH = "../../models/behavior_engine_fast.h5"
EPOCHS = 60 # Pushed slightly higher since we are controlling the learning rate
BATCH_SIZE = 32

def build_fast_temporal_model():
    inputs = Input(shape=(30, 1280))
    
    # TRICK 1: Feature Noise Injection (Data Augmentation for Arrays!)
    # This prevents the AI from memorizing exact values
    x = GaussianNoise(0.02)(inputs) 
    
    # Temporal Edge Detector
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x) 
    
    # TRICK 2: Bidirectional GRU (Looks at motion forwards and backwards)
    x = Bidirectional(GRU(32, return_sequences=False))(x)
    
    # Classification Head
    x = Dropout(0.6)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_features():
    print("Loading extracted features into RAM...")
    X, y = [], []
    classes = {'NonFight': 0, 'Fight': 1}
    
    for class_name, label in classes.items():
        class_path = os.path.join(FEATURE_DIR, class_name)
        if not os.path.exists(class_path):
            continue
            
        files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
        for feature_file in files:
            features = np.load(os.path.join(class_path, feature_file))
            X.append(features)
            y.append(label)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_features()
    if len(X) == 0:
        print("ERROR: No features loaded.")
        exit()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_fast_temporal_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    
    # TRICK 3: The Sniper Learning Rate
    # If val_loss doesn't improve for 3 epochs, cut the learning rate in half!
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    print("\n--- Starting SOTA Regularized Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print(f"\nTraining Complete! Best model saved to {MODEL_SAVE_PATH}")