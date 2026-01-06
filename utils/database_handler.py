import mysql.connector
from mysql.connector import Error
from datetime import datetime
import cv2
import numpy as np
from config import DB_CONFIG
import qrcode
from PIL import Image
import io

class DatabaseHandler:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = mysql.connector.connect(**DB_CONFIG)
            print("Database connection established")
        except Error as e:
            print(f"Error connecting to database: {e}")
    
    def save_bottle_data(self, serial_number, water_level, shape_status, confidence, bottle_image):
        try:
            if not self.connection.is_connected():
                self.connect()
            
            cursor = self.connection.cursor()
            
            # Convert image to binary
            _, buffer = cv2.imencode('.jpg', bottle_image)
            image_binary = buffer.tobytes()
            
            # Check if bottle is defective
            is_defective = (water_level in ['low', 'overflow'] or shape_status == 'defective')
            
            # Insert data
            query = """
            INSERT INTO bottles (serial_number, water_level, shape_status, confidence_score, 
                                processed_image, is_defective)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (serial_number, water_level, shape_status, confidence, image_binary, is_defective)
            
            cursor.execute(query, values)
            self.connection.commit()
            
            # Update daily statistics
            cursor.callproc('UpdateDailyStatistics')
            
            print(f"Data saved for bottle {serial_number}")
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error saving data: {e}")
            return False
    
    def get_bottle_history(self, serial_number=None, limit=50):
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if serial_number:
                query = "SELECT * FROM bottles WHERE serial_number = %s ORDER BY detection_date DESC"
                cursor.execute(query, (serial_number,))
            else:
                query = "SELECT * FROM bottles ORDER BY detection_date DESC LIMIT %s"
                cursor.execute(query, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            return results
            
        except Error as e:
            print(f"Error fetching data: {e}")
            return []
    
    def get_statistics(self):
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get today's statistics
            query = """
            SELECT 
                SUM(CASE WHEN is_defective = FALSE THEN 1 ELSE 0 END) as perfect_today,
                SUM(CASE WHEN is_defective = TRUE THEN 1 ELSE 0 END) as defective_today,
                COUNT(*) as total_today
            FROM bottles 
            WHERE DATE(detection_date) = CURDATE()
            """
            cursor.execute(query)
            today_stats = cursor.fetchone()
            
            # Get overall statistics
            query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_defective = FALSE THEN 1 ELSE 0 END) as perfect_total,
                SUM(CASE WHEN is_defective = TRUE THEN 1 ELSE 0 END) as defective_total
            FROM bottles
            """
            cursor.execute(query)
            overall_stats = cursor.fetchone()
            
            cursor.close()
            return today_stats, overall_stats
            
        except Error as e:
            print(f"Error getting statistics: {e}")
            return None, None
    
    def generate_serial_number(self):
        """Generate a unique serial number for each bottle"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        import random
        random_str = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
        return f"BTL-{timestamp}-{random_str}"
    
    def create_qr_code(self, serial_number):
        """Generate QR code for bottle serial number"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(serial_number)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        return img
    
    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")