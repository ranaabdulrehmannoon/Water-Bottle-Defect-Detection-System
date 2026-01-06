from fpdf import FPDF
from datetime import datetime
import os
from typing import List, Dict
from config import WATER_LEVEL_LABELS, SHAPE_LABELS, DATA_DIR


def bullet_list(pdf: FPDF, items, indent=10, line_height=7):
    """Render a simple bullet list."""
    max_width = pdf.w - 2 * pdf.l_margin - indent
    for item in items:
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(max_width, line_height, f"- {item}")
        pdf.ln(1)


def add_section(pdf: FPDF, title: str, paragraphs=None, bullets=None):
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    if paragraphs:
        for para in paragraphs:
            pdf.multi_cell(0, 7, para)
            pdf.ln(1)
    if bullets:
        bullet_list(pdf, bullets)
    pdf.ln(4)


# -------- Helpers --------
def read_requirements() -> List[str]:
    """Read requirements.txt in the project root if present."""
    root = os.path.dirname(os.path.abspath(__file__))
    req_path = os.path.join(root, "requirements.txt")
    if not os.path.exists(req_path):
        return []
    deps = []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            deps.append(line)
    return deps


def summarize_dataset() -> (Dict[str, int], Dict[str, int]):
    """Count images per class for water_level and shape in data/train."""
    wl_counts: Dict[str, int] = {cls: 0 for cls in WATER_LEVEL_LABELS}
    shape_counts: Dict[str, int] = {cls: 0 for cls in SHAPE_LABELS}

    train_dir = os.path.join(str(DATA_DIR), "train")
    # Water level
    wl_dir = os.path.join(train_dir, "water_level")
    for cls in WATER_LEVEL_LABELS:
        cls_dir = os.path.join(wl_dir, cls)
        wl_counts[cls] = count_images(cls_dir)

    # Shape
    shape_dir = os.path.join(train_dir, "shape")
    for cls in SHAPE_LABELS:
        cls_dir = os.path.join(shape_dir, cls)
        shape_counts[cls] = count_images(cls_dir)

    return wl_counts, shape_counts


def count_images(dir_path: str) -> int:
    """Count image files in a directory (common extensions)."""
    if not os.path.exists(dir_path):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total = 0
    for fn in os.listdir(dir_path):
        ext = os.path.splitext(fn)[1].lower()
        if ext in exts:
            total += 1
    return total


def build_report(output_path: str = "Water_Bottle_Defect_Detection_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Water Bottle Defect Detection System", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, "Comprehensive Project Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, "This report summarizes the architecture, technologies, data pipeline, models, UI, and deployment steps for the Water Bottle Defect Detection System.")
    pdf.ln(10)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")

    # Table of Contents (slide style)
    pdf.add_page()
    add_section(
        pdf,
        "Table of Contents",
        bullets=[
            "1. Problem Statement & Goals",
            "2. Technology Stack",
            "3. Data & Model Training",
            "4. Runtime Detection Pipeline",
            "5. GUI (PyQt5) Highlights",
            "6. Database Layer",
            "7. Configuration",
            "8. Operational Modes",
            "9. Usage Flow",
            "10. Deployment & Hardware",
            "11. Risks & Future Improvements",
            "12. Quick Commands",
            "13. Dataset Summary",
            "14. Features (Slide-Ready)",
            "15. Dependencies (Appendix)"
        ],
    )

    # Overview
    pdf.add_page()
    add_section(
        pdf,
        "1. Problem Statement & Goals",
        paragraphs=[
            "Automate quality control for bottled water on a production line by detecting defects in real time. The system classifies water-level issues (low, overflow) and shape defects, assigns serial numbers, stores evidence, and presents operators with a live dashboard."
        ],
        bullets=[
            "Operate with live camera input (webcam/IP stream)",
            "Classify water level and bottle shape using deep learning",
            "Highlight defects visually and in data logs",
            "Persist detections in MySQL with images, timestamps, and QR-ready serials",
            "Provide both GUI mode (PyQt5) and headless console mode"
        ],
    )

    # Tech stack
    pdf.add_page()
    add_section(
        pdf,
        "2. Technology Stack",
        bullets=[
            "Python 3 with OpenCV for camera capture, contour-based ROI detection, drawing overlays",
            "TensorFlow/Keras (MobileNetV2) for two classifiers: water level and shape",
            "PyQt5 for the operator dashboard (live video, stats, history, export)",
            "MySQL (mysql-connector-python) for persistence, history, and daily statistics",
            "Image enhancement via CLAHE; edge detection with Canny + dilation",
            "Aux libs: imutils, numpy, matplotlib (training plots), qrcode",
            "Config-driven settings: camera source, thresholds, model paths, DB credentials",
            "Packaging via requirements.txt; optional console mode for testing"
        ],
    )

    # Data and models
    pdf.add_page()
    add_section(
        pdf,
        "3. Data & Model Training",
        paragraphs=[
            "Two classifiers are trained separately using MobileNetV2 backbones with frozen base layers and custom heads. Data is expected in data/train with class folders."],
        bullets=[
            "Water level classes: full, low, overflow",
            "Shape classes: perfect, defective",
            "Preprocessing: rescale 1/255, augmentation (rotation/shift/shear/zoom/flip)",
            "Input size: 224x224, optimizer: Adam (lr=0.001), loss: categorical crossentropy",
            "Epochs: 50 in training script; adjustable via config.py",
            "Outputs saved to models/water_level_model.h5 and models/shape_model.h5",
            "Training helper: train_models.py with plot generation for accuracy/loss",
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "4. Runtime Detection Pipeline",
        bullets=[
            "Capture frame from CameraStream (webcam or IP cam)",
            "Detect bottle ROI using grayscale + Gaussian blur + Canny edges + dilation + contour area filter (min area from config)",
            "Extract padded ROI; enhance contrast using CLAHE in LAB space",
            "Resize/normalize ROI; run water-level and shape models",
            "Compute overall confidence (mean of both classifiers); gate by CONFIDENCE_THRESHOLD (0.75 by default)",
            "Generate serial number and QR-ready payload; store record + JPEG ROI in MySQL",
            "Render overlays: bounding box color-coded by defect status, status badge, class labels, confidence, serial number",
            "Cooldown between detections to avoid duplicates; maintain detection history",
            "If no bottle: show 'Scanning for bottle...'"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "5. GUI (PyQt5) Highlights",
        bullets=[
            "Main window: live video feed with overlays and status banners",
            "Controls: pause/resume detection, capture image, reset, export data",
            "StatisticsWidget: today vs overall counts (perfect vs defective)",
            "DetectionHistoryWidget: recent detections table with color-coded cells",
            "Current detection panel: serial, water level, shape, confidence, status",
            "Threaded camera loop (QThread) emitting processed frames at ~30 FPS",
            "Error handling popup if camera fails to start",
            "GUI launches only after DB connectivity check, with optional setup prompt"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "6. Database Layer",
        bullets=[
            "Connector: mysql-connector-python (DB_CONFIG in config.py)",
            "Table bottles stores serial_number, water_level, shape_status, confidence_score, processed_image (BLOB), detection_date, is_defective",
            "Stored procedure UpdateDailyStatistics updates daily aggregates",
            "APIs: save_bottle_data(), get_bottle_history(), get_statistics(), generate_serial_number(), create_qr_code()",
            "Defect rule: is_defective is True if water level is low/overflow or shape is defective",
            "Images stored as JPEG bytes for audit and export"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "7. Configuration",
        bullets=[
            "config.py centralizes camera source (webcam or IP), resolution, frame rate",
            "Model hyperparameters: IMG_SIZE, batch size, epochs, learning rate",
            "Detection knobs: CONFIDENCE_THRESHOLD (default 0.75), MIN_BOTTLE_AREA",
            "Directories ensured at startup: data/, models/, database/, logs/",
            "Visualization colors for perfect/defective and water-level states"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "8. Operational Modes",
        bullets=[
            "GUI mode (default): run python main.py",
            "Console mode: python main.py --no-gui (prints detection metadata)",
            "Database setup: python main.py --setup-db",
            "Model training: python main.py --train (delegates to train_models.py)"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "9. Usage Flow (for presentation)",
        bullets=[
            "Start application and connect to camera",
            "System scans conveyor for a bottle contour; draws magenta ROI box",
            "On valid ROI and confidence >= threshold: classify water level and shape",
            "Color-coded overlay shows status; serial assigned and saved to DB",
            "Dashboard updates today/overall stats and recent detections table",
            "Operator can pause/resume detection, capture frame, export data",
            "Database entries include stored image and can be linked to QR for traceability"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "10. Deployment & Hardware",
        bullets=[
            "Camera: USB webcam or IP camera stream (e.g., Android IP Webcam)",
            "Host: Windows PC with Python 3.9+ and GPU optional (CPU works at lower FPS)",
            "MySQL server reachable with configured credentials",
            "Models directory populated with trained .h5 files",
            "Lighting: consistent illumination improves contour detection and classifier accuracy"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "11. Risks & Future Improvements",
        bullets=[
            "Improve robustness to motion blur and lighting variance (add preprocessing, retrain)",
            "Replace contour ROI with lightweight object detector (e.g., YOLOv8n) for stability",
            "Unify models into multitask network to reduce latency",
            "Add calibration UI for thresholds and ROI masks",
            "Secure DB credentials (env vars/secrets) and add role-based access in GUI",
            "Add test automation and continuous training with new production data"
        ],
    )

    pdf.add_page()
    add_section(
        pdf,
        "12. Quick Commands",
        bullets=[
            "Run GUI: python main.py",
            "Run console test: python main.py --no-gui",
            "Train models: python main.py --train",
            "Setup DB: python main.py --setup-db"
        ],
    )

    # Dataset Summary (dynamic)
    pdf.add_page()
    wl_counts, shape_counts = summarize_dataset()
    add_section(
        pdf,
        "13. Dataset Summary",
        paragraphs=[
            "Training dataset summary gathered from data/train/. Counts reflect number of images per class found on disk.",
        ],
        bullets=[
            f"Water level classes: " + ", ".join([f"{cls}: {wl_counts.get(cls, 0)}" for cls in WATER_LEVEL_LABELS]),
            f"Shape classes: " + ", ".join([f"{cls}: {shape_counts.get(cls, 0)}" for cls in SHAPE_LABELS]),
            "Ensure balanced classes for best model generalization",
        ],
    )

    # Features (Slide-Ready)
    pdf.add_page()
    add_section(
        pdf,
        "14. Features (Slide-Ready)",
        bullets=[
            "Real-time detection at ~30 FPS with overlays",
            "Dual-model classification: water level + shape",
            "Automatic serial assignment and QR generation capability",
            "MySQL persistence with images and statistics",
            "GUI controls: pause, capture, reset, export",
            "Console mode for headless operations/testing",
            "Configurable thresholds and camera source",
            "Export-ready logs and detection history",
        ],
    )

    # Dependencies Appendix (from requirements.txt)
    pdf.add_page()
    deps = read_requirements()
    add_section(
        pdf,
        "15. Dependencies (Appendix)",
        bullets=[dep for dep in deps] if deps else ["requirements.txt not found"],
    )

    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(f"Report generated: {path}")
