import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_base_network(input_shape=(224, 224, 3)):
    """
    The 'Twin Brain' - Now powered by MobileNetV2!
    It already knows how to see objects, saving us hours of training.
    """
    # 1. Load the pre-trained expert (without its classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 2. FREEZE the expert so we don't destroy its memory!
    base_model.trainable = False 
    
    # 3. Build our custom output on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x) # Flatten the features
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    return Model(inputs, x, name="Twin_Brain")

def build_siamese_network(input_shape=(224, 224, 3)):
    left_input = Input(shape=input_shape, name="Left_Image")
    right_input = Input(shape=input_shape, name="Right_Image")
    
    base_network = build_base_network(input_shape)
    
    encoded_left = base_network(left_input)
    encoded_right = base_network(right_input)
    
    # Absolute Difference (L1 Distance) is mathematically safer than Euclidean here
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    distance = L1_layer([encoded_left, encoded_right])
    
    # Final prediction
    outputs = Dense(1, activation='sigmoid')(distance)
    
    return Model(inputs=[left_input, right_input], outputs=outputs, name="SentinAI_Threat_Detector")

if __name__ == "__main__":
    model = build_siamese_network()
    model.summary()