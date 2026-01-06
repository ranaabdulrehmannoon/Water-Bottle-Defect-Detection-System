import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from config import IMG_SIZE, WATER_LEVEL_LABELS, SHAPE_LABELS
import os

class BottleDetectorModels:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.water_level_model = None
        self.shape_model = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            water_level_path = os.path.join(self.models_dir, 'water_level_model.h5')
            shape_path = os.path.join(self.models_dir, 'shape_model.h5')
            
            if os.path.exists(water_level_path):
                self.water_level_model = load_model(water_level_path)
                print("Water level model loaded successfully")
            else:
                print(f"Water level model not found at {water_level_path}")
                
            if os.path.exists(shape_path):
                self.shape_model = load_model(shape_path)
                print("Shape model loaded successfully")
            else:
                print(f"Shape model not found at {shape_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def create_models(self):
        """Create models from scratch"""
        # Water level model
        base_model1 = MobileNetV2(weights='imagenet', include_top=False, 
                                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        
        # Freeze base model layers
        for layer in base_model1.layers:
            layer.trainable = False
        
        x = base_model1.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(WATER_LEVEL_LABELS), activation='softmax')(x)
        
        self.water_level_model = Model(inputs=base_model1.input, outputs=predictions)
        
        # Shape model
        base_model2 = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        
        for layer in base_model2.layers:
            layer.trainable = False
        
        y = base_model2.output
        y = GlobalAveragePooling2D()(y)
        y = Dense(128, activation='relu')(y)
        y = Dropout(0.5)(y)
        shape_predictions = Dense(len(SHAPE_LABELS), activation='softmax')(y)
        
        self.shape_model = Model(inputs=base_model2.input, outputs=shape_predictions)
        
        # Compile models
        self.water_level_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.shape_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Models created successfully")
    
    def train_models(self, train_data_dir, validation_split=0.2):
        """Train both models"""
        if self.water_level_model is None or self.shape_model is None:
            self.create_models()
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split,
            fill_mode='nearest'
        )
        
        # Train water level model
        water_level_train_dir = os.path.join(train_data_dir, 'water_level')
        water_level_train_generator = datagen.flow_from_directory(
            water_level_train_dir,
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        water_level_val_generator = datagen.flow_from_directory(
            water_level_train_dir,
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train shape model
        shape_train_dir = os.path.join(train_data_dir, 'shape')
        shape_train_generator = datagen.flow_from_directory(
            shape_train_dir,
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        shape_val_generator = datagen.flow_from_directory(
            shape_train_dir,
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train water level model
        print("Training water level model...")
        water_level_history = self.water_level_model.fit(
            water_level_train_generator,
            steps_per_epoch=water_level_train_generator.samples // 32,
            validation_data=water_level_val_generator,
            validation_steps=water_level_val_generator.samples // 32,
            epochs=50,
            verbose=1
        )
        
        # Train shape model
        print("Training shape model...")
        shape_history = self.shape_model.fit(
            shape_train_generator,
            steps_per_epoch=shape_train_generator.samples // 32,
            validation_data=shape_val_generator,
            validation_steps=shape_val_generator.samples // 32,
            epochs=50,
            verbose=1
        )
        
        # Save models
        os.makedirs(self.models_dir, exist_ok=True)
        self.water_level_model.save(os.path.join(self.models_dir, 'water_level_model.h5'))
        self.shape_model.save(os.path.join(self.models_dir, 'shape_model.h5'))
        
        print("Models trained and saved successfully")
        
        return water_level_history, shape_history
    
    def predict(self, image):
        """Make predictions on an image"""
        if self.water_level_model is None or self.shape_model is None:
            raise ValueError("Models not loaded or created")
        
        # Preprocess image
        image_resized = tf.image.resize(image, IMG_SIZE)
        image_normalized = image_resized / 255.0
        image_expanded = tf.expand_dims(image_normalized, axis=0)
        
        # Make predictions
        water_level_pred = self.water_level_model.predict(image_expanded, verbose=0)
        shape_pred = self.shape_model.predict(image_expanded, verbose=0)
        
        # Get labels and confidence
        water_level_idx = np.argmax(water_level_pred[0])
        water_level_label = WATER_LEVEL_LABELS[water_level_idx]
        water_level_confidence = water_level_pred[0][water_level_idx]
        
        shape_idx = np.argmax(shape_pred[0])
        shape_label = SHAPE_LABELS[shape_idx]
        shape_confidence = shape_pred[0][shape_idx]
        
        # Overall confidence
        overall_confidence = (water_level_confidence + shape_confidence) / 2
        
        return {
            'water_level': water_level_label,
            'water_level_confidence': float(water_level_confidence),
            'shape': shape_label,
            'shape_confidence': float(shape_confidence),
            'overall_confidence': float(overall_confidence)
        }