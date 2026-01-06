-- Create database
CREATE DATABASE IF NOT EXISTS bottle_defect_db;
USE bottle_defect_db;

-- Bottle information table
CREATE TABLE IF NOT EXISTS bottles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    serial_number VARCHAR(50) UNIQUE NOT NULL,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    water_level VARCHAR(20),
    shape_status VARCHAR(20),
    confidence_score FLOAT,
    image_path VARCHAR(255),
    is_defective BOOLEAN DEFAULT FALSE,
    processed_image LONGBLOB
);

-- Defect statistics table
CREATE TABLE IF NOT EXISTS defect_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    total_bottles INT DEFAULT 0,
    defective_bottles INT DEFAULT 0,
    perfect_bottles INT DEFAULT 0,
    overflow_count INT DEFAULT 0,
    low_count INT DEFAULT 0
);

-- Create indexes for better performance
CREATE INDEX idx_serial ON bottles(serial_number);
CREATE INDEX idx_date ON bottles(detection_date);
CREATE INDEX idx_defective ON bottles(is_defective);

-- Create stored procedure for daily statistics
DELIMITER $$
CREATE PROCEDURE UpdateDailyStatistics()
BEGIN
    INSERT INTO defect_statistics (date, total_bottles, defective_bottles, perfect_bottles, overflow_count, low_count)
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
END$$
DELIMITER ;