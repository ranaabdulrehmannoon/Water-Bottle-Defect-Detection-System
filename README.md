# Water Bottle Defect Detection System
## 📌 Project Overview

The Water Bottle Defect Detection System is an automated quality control solution that leverages computer vision and deep learning to identify defects in water bottles during the manufacturing process. The system performs real-time inspection using a mobile phone camera as a live video source and ensures that only defect-free bottles proceed to packaging, thereby improving product quality and operational efficiency.

## 🎯 Objectives

- Automate water bottle quality inspection

Detect water level and bottle shape defects accurately

Reduce manual inspection effort

Improve production efficiency and reliability

## ⚙️ Key Features
## 🔴 Real-Time Live Detection

Live inspection using phone camera

Real-time annotated video feed

Visual overlays indicating detected defects

## 💧 Water Level Classifier

Detects Full water levels

Identifies Low-filled bottles

Flags Overflow conditions

## 🧴 Bottle Shape Classifier

Confirms perfect bottle integrity

Detects defective or deformed shapes

## 🎨 Image Enhancement

Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for improved image quality under varying lighting conditions

## 🖥️ Graphical User Interface (GUI)

The system includes an intuitive PyQt5-based GUI that provides real-time insights and seamless control. The interface runs on a separate thread to ensure smooth performance during live detection.

## GUI Capabilities

Live Annotated Video – real-time feed with detection overlays

Status & Identification – detection status, bottle IDs, confidence scores

Statistics & History – daily/total defect counts and detailed logs

Operator Controls – pause, reset inspection, and export data

## 🧠 Technologies Used
Technology	Description
Python 3	Core programming language
OpenCV	Advanced computer vision tasks
TensorFlow / Keras	Deep learning model development
Pillow (PIL)	Image processing
NumPy	Numerical computing
Matplotlib	Data visualization
PyQt5	Interactive graphical user interface
MySQL	Robust data storage
CLAHE	Image enhancement
📊 Dataset & Model Training

Custom dataset used for training defect classifiers

Image preprocessing includes resizing, normalization, and contrast enhancement

Deep learning models trained for accurate classification of defects

## 📁 Data Management

Detection results stored in MySQL database

Supports data export for reporting and analysis

Enables traceability and historical inspection records

## 🏭 Use Cases

Bottled water manufacturing plants

Automated quality inspection systems

Academic and industrial AI research projects

## ✅ Advantages

High accuracy and consistency

Reduces human error

Scalable for industrial deployment

Real-time defect detection
