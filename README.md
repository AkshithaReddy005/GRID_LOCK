# 🚦 GRIDLOCK: Automated Traffic Violation Detection Pipeline

<div align="center">
  <h3>Built by Team Elite Hackers</h3>
  <p><b>Vasavi College of Engineering</b></p>
  <p><i>Akshitha Reddy | Shreya | Pradeep | Sandeep</i></p>
</div>

---

## 📌 Project Overview
**GRIDLOCK** is an advanced, AI-powered traffic monitoring and violation detection system. Designed for high-speed, real-time analysis of CCTV footage and static traffic images, the pipeline identifies multi-class traffic infractions with high precision. 

Rather than relying on a single monolithic model, GRIDLOCK employs a modular, multi-stage architecture combining state-of-the-art Object Detection (YOLO), custom Image Classification networks, and a deterministic Geometric Rule Engine.

## 🚀 Key Features & Capabilities

### 1. 🏍️ Helmet Non-Compliance Detection
- Extracts localized bounding boxes of riders using the detection model.
- Passes cropped Regions of Interest (ROIs) through a specialized binary classifier.
- Flags riders as `No Helmet` and dynamically recolors bounding boxes (Red for violation, Green for compliance).

### 2. 🚗 Seatbelt Violation Detection
- Detects the windshield/cabin area of cars.
- Analyzes the interior cabin crop using a highly sensitive seatbelt classification model.
- Operates efficiently even through reflections and varying lighting conditions.

### 3. 👥 Triple Riding Detection (Geometric Rule Engine)
- Bypasses traditional failure points of standard object detection by using pure mathematics.
- Calculates **Intersection over Union (IoU)** and **Bounding Box Overlap** between `person` classes and `motorcycle` classes.
- If **three or more** distinct persons have a high geometric overlap with a single motorcycle bounding box, the system instantly flags a `Triple Riding` violation.

### 4. ⚙️ Dynamic Image Preprocessing
- Integrates **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Enhances low-light, nighttime, or fog-obscured traffic camera footage before the AI inference stage to drastically improve recall.

## 🧠 System Architecture

1. **Input Stage:** High-resolution traffic images via Gradio UI or CCTV API.
2. **Preprocessing:** Optional CLAHE enhancement to normalize pixel intensity.
3. **Primary Detection (YOLO):** Detects overarching classes (`car`, `motorcycle`, `person`).
4. **Cropping & Routing:** Extracts object coordinates and routes them to specific sub-models.
5. **Secondary Classification:**
   - Rider crops $\rightarrow$ Helmet Classifier.
   - Car crops $\rightarrow$ Seatbelt Classifier.
6. **Geometric Analysis:** Person + Motorcycle boxes $\rightarrow$ Triple Riding Logic Engine.
7. **Output Stage:** Fused annotated image rendering thicker, high-visibility bounding boxes and violation tags.

## 💻 Tech Stack
- **Deep Learning Frameworks:** PyTorch, Ultralytics (YOLO11)
- **Computer Vision:** OpenCV (`cv2`)
- **Data Manipulation:** NumPy, Pandas
- **Frontend / UI:** Gradio (with custom CSS styling & responsive layout)
- **Deployment:** Hugging Face Spaces (Ready)

## 🏃‍♂️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/gridlock.git
   cd gridlock/demo_app
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Gradio App:**
   ```bash
   python app.py
   ```
   *The application will launch on `http://127.0.0.1:7860/`.*

## 🎛️ Advanced UI Controls
The Gradio Interface allows users to tweak internal pipeline parameters on the fly:
- **Helmet Classifier Confidence:** Adjust the strictness of helmet detection.
- **Seatbelt Classifier Confidence:** Fine-tune seatbelt detection thresholds.
- **Triple Riding Overlap Threshold:** Control how strictly the algorithm calculates rider-to-bike overlap.
- **NMS Threshold:** Filter overlapping bounding boxes.

---
*Built with ❤️ for safer roads. End the Gridlock.*
