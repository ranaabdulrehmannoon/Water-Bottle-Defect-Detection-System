import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Qt5Agg')

from camera_stream import CameraStream
from detector import BottleDefectDetector
from config import COLORS

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.camera = None
        self.running = False
        
    def run(self):
        self.camera = CameraStream()
        if self.camera.start():
            self.running = True
            while self.running:
                try:
                    frame = self.camera.get_frame()
                    if frame is not None:
                        processed_frame, detection_data = self.detector.process_frame(frame)
                        self.frame_ready.emit(processed_frame, detection_data)
                    self.msleep(30)  # ~30 FPS
                except Exception as e:
                    self.error_signal.emit(str(e))
                    break
            self.camera.stop()
        else:
            self.error_signal.emit("Could not start camera. Check camera connection.")
    
    def stop(self):
        self.running = False
        self.wait()

class StatisticsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Detection Statistics")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")
        layout.addWidget(title)
        
        # Stats grid
        grid = QGridLayout()
        grid.setSpacing(10)
        
        # Today's stats
        today_label = QLabel("Today's Statistics")
        today_label.setStyleSheet("font-weight: bold; color: #3498db;")
        grid.addWidget(today_label, 0, 0, 1, 2)
        
        grid.addWidget(QLabel("Total Bottles:"), 1, 0)
        self.today_total = QLabel("0")
        self.today_total.setStyleSheet("font-weight: bold;")
        grid.addWidget(self.today_total, 1, 1)
        
        grid.addWidget(QLabel("Perfect:"), 2, 0)
        self.today_perfect = QLabel("0")
        self.today_perfect.setStyleSheet("color: green; font-weight: bold;")
        grid.addWidget(self.today_perfect, 2, 1)
        
        grid.addWidget(QLabel("Defective:"), 3, 0)
        self.today_defective = QLabel("0")
        self.today_defective.setStyleSheet("color: red; font-weight: bold;")
        grid.addWidget(self.today_defective, 3, 1)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(line, 0, 2, 4, 1)
        
        # Overall stats
        overall_label = QLabel("Overall Statistics")
        overall_label.setStyleSheet("font-weight: bold; color: #3498db;")
        grid.addWidget(overall_label, 0, 3, 1, 2)
        
        grid.addWidget(QLabel("Total Bottles:"), 1, 3)
        self.overall_total = QLabel("0")
        self.overall_total.setStyleSheet("font-weight: bold;")
        grid.addWidget(self.overall_total, 1, 4)
        
        grid.addWidget(QLabel("Perfect:"), 2, 3)
        self.overall_perfect = QLabel("0")
        self.overall_perfect.setStyleSheet("color: green; font-weight: bold;")
        grid.addWidget(self.overall_perfect, 2, 4)
        
        grid.addWidget(QLabel("Defective:"), 3, 3)
        self.overall_defective = QLabel("0")
        self.overall_defective.setStyleSheet("color: red; font-weight: bold;")
        grid.addWidget(self.overall_defective, 3, 4)
        
        layout.addLayout(grid)
        self.setLayout(layout)
    
    def update_stats(self, today_stats, overall_stats):
        if today_stats:
            self.today_total.setText(str(today_stats.get('total_today', 0) or 0))
            self.today_perfect.setText(str(today_stats.get('perfect_today', 0) or 0))
            self.today_defective.setText(str(today_stats.get('defective_today', 0) or 0))
        
        if overall_stats:
            self.overall_total.setText(str(overall_stats.get('total', 0) or 0))
            self.overall_perfect.setText(str(overall_stats.get('perfect_total', 0) or 0))
            self.overall_defective.setText(str(overall_stats.get('defective_total', 0) or 0))

class DetectionHistoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Recent Detections")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #2c3e50;")
        layout.addWidget(title)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Serial", "Water Level", "Shape", "Confidence"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #f8f9fa;
                alternate-background-color: #e9ecef;
                gridline-color: #dee2e6;
            }
            QHeaderView::section {
                background-color: #007bff;
                color: white;
                padding: 4px;
                border: 1px solid #6c757d;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(self.table)
        self.setLayout(layout)
    
    def update_history(self, history):
        self.table.setRowCount(0)
        
        for i, record in enumerate(history):
            self.table.insertRow(i)
            
            # Time
            time_str = record['detection_date'].strftime("%H:%M:%S") if isinstance(record['detection_date'], datetime) else str(record['detection_date'])
            time_item = QTableWidgetItem(time_str)
            
            # Serial
            serial_item = QTableWidgetItem(record.get('serial_number', 'N/A'))
            
            # Water level with color coding
            water_level = record.get('water_level', 'N/A')
            water_item = QTableWidgetItem(water_level)
            if water_level == 'overflow':
                water_item.setBackground(QColor(*COLORS['overflow']))
                water_item.setForeground(QColor(255, 255, 255))
            elif water_level == 'low':
                water_item.setBackground(QColor(*COLORS['low']))
            else:
                water_item.setBackground(QColor(*COLORS['full']))
            
            # Shape with color coding
            shape_status = record.get('shape_status', 'N/A')
            shape_item = QTableWidgetItem(shape_status)
            if shape_status == 'defective':
                shape_item.setBackground(QColor(*COLORS['defective']))
                shape_item.setForeground(QColor(255, 255, 255))
            else:
                shape_item.setBackground(QColor(*COLORS['perfect']))
            
            # Confidence
            confidence = record.get('confidence_score', 0)
            conf_item = QTableWidgetItem(f"{confidence:.2%}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Add items to table
            self.table.setItem(i, 0, time_item)
            self.table.setItem(i, 1, serial_item)
            self.table.setItem(i, 2, water_item)
            self.table.setItem(i, 3, shape_item)
            self.table.setItem(i, 4, conf_item)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = BottleDefectDetector()
        self.video_thread = None
        self.detection_enabled = True
        self.init_ui()
        self.start_camera()
        
    def init_ui(self):
        self.setWindowTitle("🚰 Water Bottle Defect Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set window icon
        self.setWindowIcon(QIcon(self.create_icon()))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Video feed (60%)
        left_panel = QVBoxLayout()
        
        # Video display with frame
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        video_frame.setLineWidth(2)
        video_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            border: 2px solid #ccc;
            background-color: #1a1a1a;
            color: white;
        """)
        self.video_label.setText("Initializing camera...")
        
        video_layout.addWidget(self.video_label)
        video_frame.setLayout(video_layout)
        left_panel.addWidget(video_frame)
        
        # Status label
        self.status_label = QLabel("🔍 Ready - Waiting for bottle detection...")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            padding: 8px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-top: 5px;
        """)
        left_panel.addWidget(self.status_label)
        
        # Control buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("⏸️ Pause Detection")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.start_btn.clicked.connect(self.toggle_detection)
        
        self.capture_btn = QPushButton("📸 Capture Image")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.capture_btn.clicked.connect(self.capture_image)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_detection)
        
        self.export_btn = QPushButton("💾 Export Data")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.export_btn.clicked.connect(self.export_data)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.export_btn)
        button_frame.setLayout(button_layout)
        left_panel.addWidget(button_frame)
        
        main_layout.addLayout(left_panel, 60)
        
        # Right panel - Information (40%)
        right_panel = QVBoxLayout()
        
        # Statistics widget
        self.stats_widget = StatisticsWidget()
        self.stats_widget.setMaximumHeight(180)
        right_panel.addWidget(self.stats_widget)
        
        # Current detection info
        self.current_info_group = QGroupBox("📊 Current Detection")
        self.current_info_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #6c757d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        current_info_layout = QFormLayout()
        current_info_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.serial_label = QLabel("N/A")
        self.serial_label.setStyleSheet("font-weight: bold; color: #007bff;")
        
        self.water_level_label = QLabel("N/A")
        self.water_level_label.setStyleSheet("font-weight: bold;")
        
        self.shape_label = QLabel("N/A")
        self.shape_label.setStyleSheet("font-weight: bold;")
        
        self.confidence_label = QLabel("N/A")
        self.confidence_label.setStyleSheet("font-weight: bold;")
        
        self.status_label_det = QLabel("N/A")
        self.status_label_det.setStyleSheet("font-weight: bold;")
        
        current_info_layout.addRow("Serial Number:", self.serial_label)
        current_info_layout.addRow("Water Level:", self.water_level_label)
        current_info_layout.addRow("Shape:", self.shape_label)
        current_info_layout.addRow("Confidence:", self.confidence_label)
        current_info_layout.addRow("Status:", self.status_label_det)
        
        self.current_info_group.setLayout(current_info_layout)
        right_panel.addWidget(self.current_info_group)
        
        # Detection history
        self.history_widget = DetectionHistoryWidget()
        right_panel.addWidget(self.history_widget)
        
        main_layout.addLayout(right_panel, 40)
        
        # Timer for updating stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_statistics)
        self.stats_timer.start(3000)  # Update every 3 seconds
        
        # Update stats immediately
        self.update_statistics()
        
        # Status bar
        self.statusBar().showMessage("System Ready | MySQL Connected | Camera Active")
        
        # Menu bar
        self.create_menu_bar()
    
    def create_icon(self):
        # Create a simple icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(0, 123, 255))
        painter.setPen(QPen(QColor(0, 86, 179), 2))
        painter.drawEllipse(4, 4, 24, 24)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(12, 12, 8, 8)
        painter.end()
        return QIcon(pixmap)
    
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        export_action = QAction('&Export Data', self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('&Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        show_stats_action = QAction('&Show Statistics', self, checkable=True, checked=True)
        show_stats_action.toggled.connect(self.toggle_statistics)
        view_menu.addAction(show_stats_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def start_camera(self):
        self.video_thread = VideoThread(self.detector)
        self.video_thread.frame_ready.connect(self.update_video)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.start()
    
    def update_video(self, frame, detection_data):
        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                     Qt.AspectRatioMode.KeepAspectRatio, 
                                     Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update current detection info
        if detection_data:
            self.serial_label.setText(detection_data['serial'])
            self.water_level_label.setText(detection_data['water_level'])
            self.shape_label.setText(detection_data['shape'])
            self.confidence_label.setText(f"{detection_data['confidence']:.2%}")
            
            is_defective = (detection_data['water_level'] in ['low', 'overflow'] or 
                           detection_data['shape'] == 'defective')
            
            if is_defective:
                status = "❌ DEFECTIVE"
                self.status_label_det.setText(status)
                self.status_label_det.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
                self.status_label.setText("🚨 Defective bottle detected!")
                self.status_label.setStyleSheet("""
                    font-size: 14px;
                    padding: 8px;
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 5px;
                    color: #721c24;
                    margin-top: 5px;
                """)
            else:
                status = "✅ PERFECT"
                self.status_label_det.setText(status)
                self.status_label_det.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
                self.status_label.setText("✓ Perfect bottle detected")
                self.status_label.setStyleSheet("""
                    font-size: 14px;
                    padding: 8px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 5px;
                    color: #155724;
                    margin-top: 5px;
                """)
            
            # Update status bar
            self.statusBar().showMessage(f"Last detection: {detection_data['serial']} | {status}")
    
    def update_statistics(self):
        try:
            today_stats, overall_stats = self.detector.get_statistics()
            self.stats_widget.update_stats(today_stats, overall_stats)
            
            # Update history
            history = self.detector.get_recent_detections(10)
            self.history_widget.update_history(history)
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.start_btn.setText("⏸️ Pause Detection")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            self.status_label.setText("▶️ Detection resumed")
        else:
            self.start_btn.setText("▶️ Resume Detection")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            self.status_label.setText("⏸️ Detection paused")
    
    def capture_image(self):
        from datetime import datetime
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Get current frame from video label
        pixmap = self.video_label.pixmap()
        if pixmap:
            pixmap.save(filename)
            self.status_label.setText(f"📸 Image saved as {filename}")
            QMessageBox.information(self, "Image Saved", f"Image saved as {filename}")
    
    def reset_detection(self):
        self.detector.reset_detection()
        self.serial_label.setText("N/A")
        self.water_level_label.setText("N/A")
        self.shape_label.setText("N/A")
        self.confidence_label.setText("N/A")
        self.status_label_det.setText("N/A")
        self.status_label_det.setStyleSheet("")
        self.status_label.setText("🔄 Reset - Ready for new detection")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            padding: 8px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            color: #856404;
            margin-top: 5px;
        """)
    
    def export_data(self):
        from datetime import datetime
        import csv
        
        filename = f"bottle_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            history = self.detector.get_recent_detections(1000)  # Get all records
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['serial_number', 'detection_date', 'water_level', 
                             'shape_status', 'confidence_score', 'is_defective']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in history:
                    writer.writerow({
                        'serial_number': record['serial_number'],
                        'detection_date': record['detection_date'],
                        'water_level': record['water_level'],
                        'shape_status': record['shape_status'],
                        'confidence_score': record['confidence_score'],
                        'is_defective': record['is_defective']
                    })
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Data exported to {filename}\n\nTotal records: {len(history)}")
        
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error exporting data: {str(e)}")
    
    def toggle_statistics(self, visible):
        self.stats_widget.setVisible(visible)
    
    def show_about(self):
        QMessageBox.about(self, "About Water Bottle Defect Detection System",
                         """<h3>Water Bottle Defect Detection System</h3>
                         <p>Version 1.0</p>
                         <p>This system detects water level and shape defects in bottles using AI.</p>
                         <p>Features:</p>
                         <ul>
                             <li>Real-time bottle detection</li>
                             <li>Water level classification (Low/Full/Overflow)</li>
                             <li>Shape defect detection</li>
                             <li>MySQL database integration</li>
                             <li>Professional GUI with statistics</li>
                         </ul>
                         <p>© 2024 Bottle Detection System</p>""")
    
    def show_error(self, error_msg):
        QMessageBox.critical(self, "Camera Error", error_msg)
        self.status_label.setText("❌ Camera Error - Check connection")
    
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        self.detector.close()
        
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application palette for better look
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.Button, QColor(248, 249, 250))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 123, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()