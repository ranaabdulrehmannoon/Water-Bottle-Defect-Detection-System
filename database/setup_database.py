import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG
import os

def setup_database():
    print("Setting up database...")
    
    try:
        # Connect to MySQL without specifying database
        config = DB_CONFIG.copy()
        database_name = config.pop('database')
        
        # First connect without database to create it
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        print(f"Database '{database_name}' created or already exists")
        
        # Use the database
        cursor.execute(f"USE {database_name}")
        
        # Bottle information table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bottles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            serial_number VARCHAR(50) UNIQUE NOT NULL,
            detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            water_level VARCHAR(20),
            shape_status VARCHAR(20),
            confidence_score FLOAT,
            image_path VARCHAR(255),
            is_defective BOOLEAN DEFAULT FALSE,
            processed_image LONGBLOB,
            INDEX idx_serial (serial_number),
            INDEX idx_date (detection_date),
            INDEX idx_defective (is_defective)
        )
        """)
        print("Table 'bottles' created or already exists")
        
        # Defect statistics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_statistics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            total_bottles INT DEFAULT 0,
            defective_bottles INT DEFAULT 0,
            perfect_bottles INT DEFAULT 0,
            overflow_count INT DEFAULT 0,
            low_count INT DEFAULT 0,
            INDEX idx_date (date)
        )
        """)
        print("Table 'defect_statistics' created or already exists")
        
        # Create stored procedure for daily statistics
        cursor.execute("DROP PROCEDURE IF EXISTS UpdateDailyStatistics")
        
        create_procedure_sql = """
        CREATE PROCEDURE UpdateDailyStatistics()
        BEGIN
            INSERT INTO defect_statistics (date, total_bottles, defective_bottles, 
                                          perfect_bottles, overflow_count, low_count)
            SELECT 
                DATE(detection_date) as date,
                COUNT(*) as total_bottles,
                SUM(CASE WHEN is_defective = TRUE THEN 1 ELSE 0 END) as defective_bottles,
                SUM(CASE WHEN is_defective = FALSE THEN 1 ELSE 0 END) as perfect_bottles,
                SUM(CASE WHEN water_level = 'overflow' THEN 1 ELSE 0 END) as overflow_count,
                SUM(CASE WHEN water_level = 'low' THEN 1 ELSE 0 END) as low_count
            FROM bottles
            WHERE DATE(detection_date) = CURDATE()
            GROUP BY DATE(detection_date)
            ON DUPLICATE KEY UPDATE
                total_bottles = VALUES(total_bottles),
                defective_bottles = VALUES(defective_bottles),
                perfect_bottles = VALUES(perfect_bottles),
                overflow_count = VALUES(overflow_count),
                low_count = VALUES(low_count);
        END
        """
        
        cursor.execute(create_procedure_sql)
        print("Stored procedure 'UpdateDailyStatistics' created")
        
        # Insert some sample data for testing
        try:
            cursor.execute("""
            INSERT INTO bottles (serial_number, water_level, shape_status, confidence_score, is_defective)
            VALUES 
                ('BTL-TEST-001', 'full', 'perfect', 0.95, FALSE),
                ('BTL-TEST-002', 'overflow', 'perfect', 0.88, TRUE),
                ('BTL-TEST-003', 'low', 'defective', 0.92, TRUE)
            ON DUPLICATE KEY UPDATE detection_date = CURRENT_TIMESTAMP
            """)
            print("Sample data inserted")
        except:
            print("Sample data already exists or error inserting")
        
        connection.commit()
        
        print("\nDatabase setup completed successfully!")
        print(f"Database: {database_name}")
        print("Tables created: bottles, defect_statistics")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"Error setting up database: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure MySQL is running")
        print("2. Check your password in config.py")
        print("3. Verify MySQL user has proper privileges")
        print("4. Try connecting with MySQL Workbench first")

if __name__ == "__main__":
    setup_database()