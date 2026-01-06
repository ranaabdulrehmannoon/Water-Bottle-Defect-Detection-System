import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import BottleDetectorModels
from config import DATA_DIR
import matplotlib.pyplot as plt

def plot_training_history(history, title):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_training.png')
    plt.show()

def main():
    print("Starting model training...")
    
    # Initialize models
    detector = BottleDetectorModels()
    
    # Train models
    train_data_dir = DATA_DIR / "train"
    
    if not os.path.exists(train_data_dir):
        print(f"Training data directory not found: {train_data_dir}")
        print("Please organize your images in the following structure:")
        print("data/train/water_level/[full, overflow, low]/")
        print("data/train/shape/[perfect, defective]/")
        return
    
    print("Training water level model...")
    water_history, shape_history = detector.train_models(str(train_data_dir))
    
    # Plot training history
    plot_training_history(water_history, "Water Level Model")
    plot_training_history(shape_history, "Shape Model")
    
    print("Training completed successfully!")
    print("Models saved in 'models/' directory")

if __name__ == "__main__":
    main()