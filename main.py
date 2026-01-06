import sys
import os
import argparse
from database.setup_database import setup_database


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'opencv-python',
        'numpy',
        'tensorflow',
        'torch',
        'torchvision',
        'scikit-learn',
        'pillow',
        'PyQt5',
        'mysql-connector-python',
        'imutils',
        'matplotlib',
        'qrcode'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def main():
    parser = argparse.ArgumentParser(description='Water Bottle Defect Detection System')
    parser.add_argument('--setup-db', action='store_true', help='Set up database')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI (for testing)')

    args = parser.parse_args()

    # ✅ Allow database setup WITHOUT PyQt6
    if args.setup_db:
        setup_database()
        return

    # ✅ Allow training WITHOUT PyQt6
    if args.train:
        from train_models import train_models
        train_models()
        return

    # ✅ Console mode WITHOUT PyQt6
    if args.no_gui:
        print("Running in console mode...")
        from detector import BottleDefectDetector
        from camera_stream import CameraStream
        import cv2

        detector = BottleDefectDetector()
        camera = CameraStream()

        if camera.start():
            print("Camera started. Press 'q' to quit, 's' to save current frame")
            try:
                while True:
                    frame = camera.get_frame()
                    if frame is not None:
                        processed_frame, detection_data = detector.process_frame(frame)

                        if detection_data:
                            print(f"\nDetection: {detection_data['serial']}")
                            print(f"Water Level: {detection_data['water_level']}")
                            print(f"Shape: {detection_data['shape']}")
                            print(f"Confidence: {detection_data['confidence']:.2%}")

                        cv2.imshow('Bottle Detection', processed_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            cv2.imwrite('capture.jpg', frame)
                            print("Frame saved as 'capture.jpg'")
            finally:
                camera.stop()
                cv2.destroyAllWindows()
                detector.close()
        return

    # ✅ GUI MODE ONLY (Now PyQt5 loads safely here)
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from gui import MainWindow

    app = QApplication(sys.argv)

    # Database check
    try:
        import mysql.connector
        from config import DB_CONFIG
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.close()
    except Exception as e:
        reply = QMessageBox.question(
            None,
            'Database Connection Error',
            f'Could not connect to database:\n{str(e)}\n\nWould you like to set up the database now?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            setup_database()
        else:
            QMessageBox.warning(
                None,
                'Warning',
                'Running without database. Some features may not work properly.'
            )

    app.setStyle('Fusion')
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
