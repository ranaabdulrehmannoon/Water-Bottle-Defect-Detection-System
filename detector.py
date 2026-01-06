import cv2
import numpy as np
import time
from datetime import datetime
from utils.model_loader import BottleDetectorModels
from utils.image_processing import ImageProcessor
from utils.database_handler import DatabaseHandler
from config import CONFIDENCE_THRESHOLD

class BottleDefectDetector:
    def __init__(self):
        self.models = BottleDetectorModels()
        self.image_processor = ImageProcessor()
        self.database = DatabaseHandler()
        self.current_serial = None
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds between detections
        self.detection_history = []
        
    def process_frame(self, frame):
        """Process a single frame for bottle detection"""
        # Create a copy for display
        display_frame = frame.copy()
        
        # Detect bottle in frame
        bottle_roi, bbox, contour = self.image_processor.detect_bottle(frame)
        
        if bottle_roi is not None and bbox is not None:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_cooldown:
                # Enhance image
                enhanced_roi = self.image_processor.enhance_image(bottle_roi)
                
                # Make predictions
                predictions = self.models.predict(enhanced_roi)
                
                # Check confidence
                if predictions['overall_confidence'] > CONFIDENCE_THRESHOLD:
                    # Generate serial number
                    self.current_serial = self.database.generate_serial_number()
                    
                    # Save to database
                    success = self.database.save_bottle_data(
                        self.current_serial,
                        predictions['water_level'],
                        predictions['shape'],
                        predictions['overall_confidence'],
                        enhanced_roi
                    )
                    
                    if success:
                        # Add to history
                        detection_data = {
                            'timestamp': datetime.now(),
                            'serial': self.current_serial,
                            'water_level': predictions['water_level'],
                            'shape': predictions['shape'],
                            'confidence': predictions['overall_confidence'],
                            'bbox': bbox
                        }
                        self.detection_history.append(detection_data)
                        
                        # Update last detection time
                        self.last_detection_time = current_time
                        
                        # Draw detection info
                        display_frame = self.image_processor.draw_detection_info(
                            display_frame, bbox,
                            predictions['water_level'],
                            predictions['shape'],
                            predictions['overall_confidence'],
                            self.current_serial
                        )
                        
                        # Draw contour
                        if contour is not None:
                            cv2.drawContours(display_frame, [contour], -1, (0, 255, 255), 2)
                        
                        return display_frame, detection_data
        
        # Draw "Scanning..." text if no bottle detected
        cv2.putText(display_frame, "Scanning for bottle...", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display_frame, None
    
    def get_statistics(self):
        """Get detection statistics"""
        return self.database.get_statistics()
    
    def get_recent_detections(self, limit=10):
        """Get recent detections from database"""
        return self.database.get_bottle_history(limit=limit)
    
    def reset_detection(self):
        """Reset current detection"""
        self.current_serial = None
    
    def close(self):
        """Close detector resources"""
        self.database.close()