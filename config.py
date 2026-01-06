import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, DATABASE_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Camera settings
CAMERA_SOURCE = "http://10.7.240.13:8080/video"  # 0 for webcam, or IP for phone camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_RATE = 30

# Model settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Reduced for faster training
LEARNING_RATE = 0.001

# Detection settings
CONFIDENCE_THRESHOLD = 0.75  # Lowered threshold for better detection
MIN_BOTTLE_AREA = 3000  # Reduced minimum area for bottle detection

# Database settings - UPDATED WITH YOUR PASSWORD
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Thinkpadtesla@79',  # Your password here
    'database': 'bottle_defect_db',
    'auth_plugin': 'mysql_native_password'  # Added for MySQL 8
}

# Classification labels
WATER_LEVEL_LABELS = ['low', 'full', 'overflow']
SHAPE_LABELS = ['perfect', 'defective']

# Colors for visualization
COLORS = {
    'perfect': (0, 255, 0),      # Green
    'defective': (0, 0, 255),    # Red
    'full': (0, 255, 0),         # Green
    'low': (0, 165, 255),        # Orange
    'overflow': (0, 0, 255),     # Red
    'scanning': (255, 255, 0)    # Yellow
}