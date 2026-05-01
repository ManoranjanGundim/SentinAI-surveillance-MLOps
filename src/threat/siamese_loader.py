import os
import cv2
import numpy as np
import random
import tensorflow as tf

class SiameseDataGenerator(tf.keras.utils.Sequence):
    # FIX 1: Added **kwargs and super().__init__ for modern Keras
    def __init__(self, data_dir, batch_size=16, img_size=(224, 224), **kwargs):
        super().__init__(**kwargs) 
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.image_paths = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths[cls].append(os.path.join(cls_dir, img_name))
                    
        self.classes = [cls for cls in self.classes if len(self.image_paths[cls]) >= 2]
        print(f"✅ Siamese Loader initialized with {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return 50 

    def read_img(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        return img / 255.0 

    def __getitem__(self, idx):
        pairs = []
        labels = []
        
        for _ in range(self.batch_size // 2):
            
            # 1. POSITIVE PAIR (Same Class)
            cls = random.choice(self.classes)
            img1_path, img2_path = random.sample(self.image_paths[cls], 2)
            pairs.append([self.read_img(img1_path), self.read_img(img2_path)])
            labels.append(1.0) 
            
            # 2. NEGATIVE PAIR (Different Classes)
            cls1, cls2 = random.sample(self.classes, 2)
            img1_path = random.choice(self.image_paths[cls1])
            img2_path = random.choice(self.image_paths[cls2])
            pairs.append([self.read_img(img1_path), self.read_img(img2_path)])
            labels.append(0.0) 

        pairs = np.array(pairs)
        labels = np.array(labels)
        
        # FIX 2: Return a Dictionary mapping perfectly to our Siamese Model inputs!
        inputs = {
            "Left_Image": pairs[:, 0],
            "Right_Image": pairs[:, 1]
        }
        
        # We must return a tuple of (inputs, labels)
        return (inputs, labels)

if __name__ == "__main__":
    TEST_DIR = "../../data/threat"
    gen = SiameseDataGenerator(TEST_DIR, batch_size=4)
    inputs, labels = gen[0]
    print(f"Success! Generated a batch.")