import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Dropout, GRU, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.models import Model

def build_sentinai_behavior_model(sequence_length=30, img_size=224):
    input_shape = (sequence_length, img_size, img_size, 3)
    inputs = Input(shape=input_shape)
    
    # 1. Spatial Feature Extractor (EfficientNet-B0)
    # Frozen to prevent your computer from crashing
    cnn_base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    cnn_base.trainable = False 
    
    encoded_frames = TimeDistributed(cnn_base)(inputs) # Output: (Batch, 30, 1280)
    
    # 2. NOVEL TEMPORAL ENGINE: Temporal 1D-CNN
    # This acts as a "Motion Edge Detector" across time
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(encoded_frames)
    x = BatchNormalization()(x)
    
    # Compress the timeline: Drops sequence from 30 down to 15, drastically saving memory!
    x = MaxPooling1D(pool_size=2)(x) 
    
    # 3. Sequence Modeler: GRU
    
    x = GRU(64, return_sequences=False)(x)
    
    # 4. Classification Head
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=inputs, outputs=outputs, name="SentinAI_Temporal_ConvGRU")
    return model

if __name__ == "__main__":
    model = build_sentinai_behavior_model()
    model.summary()