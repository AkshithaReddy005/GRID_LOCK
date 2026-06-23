---
title: Gridlock Traffic AI
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🚦 GRIDLOCK 2.0: Automated Traffic Violation Detection Pipeline

<div align="center">
  <img src="architecture_grid_lock.jpeg" alt="System Architecture" width="800"/>
  <h3>🏆 Round 2 Submission | Gridlock Hackathon 2.0</h3>
  <p><b>Built by Team Elite Hackers</b></p>
  <p><b>Vasavi College of Engineering</b></p>
  <p><i>Akshitha Reddy | Shreya | Pradeep | Sandeep</i></p>
</div>

---

## 📌 Project Overview
**GRIDLOCK 2.0** is an enterprise-grade, AI-powered traffic monitoring and violation detection system. Designed for high-speed, real-time analysis of CCTV footage and static traffic images, the pipeline identifies multi-class traffic infractions with exceptional precision. 

Moving beyond standard single-model detection, GRIDLOCK 2.0 employs an advanced, multi-stage architecture. We combine the state-of-the-art Object Detection (YOLOv8) backbone with lightweight custom Convolutional Neural Networks (EfficientNet & MobileNetV3), a deterministic Geometric Rule Engine, and an OCR module to create a fully comprehensive enforcement system.

## 🚀 Key Features & Capabilities

### 1. 🏍️ Helmet Non-Compliance Detection
- Extracts localized bounding boxes of riders using the YOLOv8 primary detector.
- Passes cropped Regions of Interest (ROIs) through our custom-trained **EfficientNet classifier**.
- Instantly flags `No Helmet` violations and dynamically recolors bounding boxes (Red for violation, Green for compliance).

### 2. 🚗 Seatbelt Violation Detection
- Detects the windshield and cabin area of vehicles.
- Analyzes the interior crop using our highly sensitive **MobileNetV3 classifier**.
- Operates robustly through windshield reflections and varying ambient lighting conditions.

### 3. 👥 Triple Riding Detection (Geometric Rule Engine)
- Bypasses traditional failure points of standard AI by employing pure coordinate mathematics.
- Calculates **Intersection over Union (IoU)** and **Bounding Box Overlap** between `person` classes and `motorcycle` classes.
- If **three or more** distinct persons have a >60% geometric overlap with a single motorcycle, the system flags a `Triple Riding` violation.

### 4. 🔤 License Plate OCR (PaddleOCR)
- Uses high-performance **PaddleOCR** to automatically and accurately extract text from captured license plates.
- Prepares structured JSON evidence payloads for seamless integration with RTO/DMV automated e-Challan systems.

### 5. ⚙️ Dynamic Image Preprocessing
- Integrates **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Enhances low-light, nighttime, or fog-obscured traffic camera footage *before* the AI inference stage to drastically improve model recall in poor weather conditions.

## 🧠 System Architecture & Workflow

1. **Input Stage:** High-resolution traffic images via the cinematic Gradio UI.
2. **Preprocessing:** CLAHE enhancement to normalize pixel intensity.
3. **Primary Detection (YOLOv8):** Detects overarching classes (`car`, `motorcycle`, `person`).
4. **Cropping & Routing:** Extracts object coordinates and routes them dynamically to specific sub-models.
5. **Secondary Classification:**
   - Rider crops $\rightarrow$ EfficientNet Helmet Classifier.
   - Car crops $\rightarrow$ MobileNetV3 Seatbelt Classifier.
6. **Geometric Analysis:** Person + Motorcycle boxes $\rightarrow$ Triple Riding Logic Engine.
7. **OCR Extraction:** Plate boxes $\rightarrow$ PaddleOCR Engine.
8. **Output Stage:** A fused, fully-annotated image rendering high-visibility bounding boxes alongside a structured violation report.

## 💻 Tech Stack
- **Deep Learning Frameworks:** PyTorch, Ultralytics (YOLOv8)
- **Custom Classifiers:** EfficientNet, MobileNetV3
- **OCR Engine:** PaddleOCR
- **Computer Vision & Image Processing:** OpenCV (`cv2`)
- **Data Manipulation:** NumPy, Pandas
- **Frontend / UI:** Gradio (with custom CSS "frosted glass" styling & responsive layout)
- **Deployment:** Ready for Hugging Face Spaces & Edge (TensorRT)

## 🏃‍♂️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AkshithaReddy005/GRID_LOCK.git
   cd GRID_LOCK
   ```

2. **Set Up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   *Make sure all `.pt` model weights are in the directory.*
   ```bash
   python app.py
   ```
   *The application will launch locally. Open your browser and go to `http://127.0.0.1:7860/`.*

## 🎛️ Advanced UI Controls
Our custom Gradio Interface allows reviewers and traffic operators to tweak internal pipeline parameters on the fly:
- **Helmet Classifier Confidence:** Adjust the strictness of helmet detection.
- **Seatbelt Classifier Confidence:** Fine-tune seatbelt detection thresholds.
- **Triple Riding Overlap Threshold:** Control how strictly the algorithm calculates rider-to-bike coordinate overlap.
- **NMS Threshold:** Filter overlapping bounding boxes for cleaner outputs.

---
*Built with ❤️ for safer roads. End the Gridlock.*
