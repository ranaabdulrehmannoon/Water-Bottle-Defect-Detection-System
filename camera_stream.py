import cv2
import threading
import time
from queue import Queue
from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT

class CameraStream:
    def __init__(self, source=CAMERA_SOURCE, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.frame_queue = Queue(maxsize=1)
        
    def start(self):
        """Start camera stream"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera source {self.source}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
        
        # Wait for first frame
        time.sleep(1)
        return True
    
    def _update_frame(self):
        """Continuously capture frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                if self.frame_queue.empty():
                    try:
                        self.frame_queue.put(frame.copy(), block=False)
                    except:
                        pass
            else:
                print("Error: Could not read frame")
                break
            time.sleep(0.03)  # ~30 FPS
    
    def get_frame(self):
        """Get current frame"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None if self.frame is None else self.frame.copy()
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def stop(self):
        """Stop camera stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def release(self):
        """Release camera resources"""
        self.stop()
    
    def get_camera_info(self):
        """Get camera information"""
        if self.cap:
            return {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'brightness': int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS)),
                'contrast': int(self.cap.get(cv2.CAP_PROP_CONTRAST))
            }
        return {}