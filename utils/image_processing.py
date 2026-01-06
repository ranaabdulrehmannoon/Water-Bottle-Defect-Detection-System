import cv2
import numpy as np
from imutils import contours
import imutils
from config import MIN_BOTTLE_AREA, COLORS

class ImageProcessor:
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)
    
    def detect_bottle(self, frame):
        """Detect bottle in the frame and extract ROI"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to close gaps
        dilated = cv2.dilate(edges, self.kernel, iterations=2)
        
        # Find contours
        cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) > 0:
            # Sort contours by area
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            
            for contour in cnts:
                area = cv2.contourArea(contour)
                
                if area > MIN_BOTTLE_AREA:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract ROI with padding
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    bottle_roi = frame[y1:y2, x1:x2]
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    return bottle_roi, (x1, y1, x2 - x1, y2 - y1), contour
        
        return None, None, None
    
    def preprocess_for_model(self, image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def enhance_image(self, image):
        """Enhance image quality for better detection"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def draw_detection_info(self, frame, bbox, water_level, shape_status, confidence, serial_number):
        """Draw detection information on frame"""
        x, y, w, h = bbox
        
        # Draw bounding box with status color
        status_color = COLORS['defective'] if (water_level in ['low', 'overflow'] or shape_status == 'defective') else COLORS['perfect']
        cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 3)
        
        # Create info text
        info_text = f"Level: {water_level} | Shape: {shape_status} | Conf: {confidence:.2f}"
        
        # Draw background for text
        cv2.rectangle(frame, (x, y - 60), (x + 400, y), status_color, -1)
        cv2.rectangle(frame, (x, y - 60), (x + 400, y), status_color, 2)
        
        # Draw text
        cv2.putText(frame, f"Serial: {serial_number}", (x + 5, y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (x + 5, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw status indicator
        status = "DEFECTIVE" if (water_level in ['low', 'overflow'] or shape_status == 'defective') else "PERFECT"
        status_bg = (0, 0, 255) if status == "DEFECTIVE" else (0, 255, 0)
        
        cv2.rectangle(frame, (x + w - 120, y), (x + w, y + 40), status_bg, -1)
        cv2.putText(frame, status, (x + w - 110, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def extract_water_level_region(self, bottle_image):
        """Extract region of interest for water level detection"""
        h, w = bottle_image.shape[:2]
        
        # Define ROI for water level (middle 60% of bottle)
        roi_height = int(h * 0.6)
        start_y = int(h * 0.2)
        
        water_roi = bottle_image[start_y:start_y + roi_height, int(w*0.3):int(w*0.7)]
        
        return water_roi