import gradio as gr
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import os
import joblib
import torchvision.transforms as T
import torch.nn as nn
from torchvision import models
from datetime import datetime

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# Force CPU for Hugging Face free tier CPU spaces
DEVICE = torch.device('cpu')

# ─── 1. MODEL ARCHITECTURE DEFINITIONS ───
def build_efficientnet_b3(num_classes, dropout=0.4):
    model = models.efficientnet_b3(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_f, num_classes)
    )
    return model

def build_mobilenet_large(num_classes, dropout=0.3):
    model = models.mobilenet_v3_large(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return model

# Class names maps
HELMET_CLASSES    = ['WithHelmet', 'WithoutHelmet']
SEATBELT_CLASSES  = ['NoSeatbelt', 'Seatbelt']

# Helper functions for the Rule Engine
def _bbox_area(b):
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

def _intersection_area(a, b):
    x_left, y_top = max(a[0], b[0]), max(a[1], b[1])
    x_right, y_bottom = min(a[2], b[2]), min(a[3], b[3])
    if x_right > x_left and y_bottom > y_top:
        return (x_right - x_left) * (y_bottom - y_top)
    return 0.0

def _calculate_iou(a, b):
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0

def _expand_box(b, frac, max_w, max_h):
    w = b[2] - b[0]
    h = b[3] - b[1]
    dx = w * frac
    
    # Aggressively expand UPWARDS to catch floating rider torso boxes
    dy_up = h * max(frac, 0.80) 
    dy_down = h * frac
    
    return [max(0, b[0] - dx), max(0, b[1] - dy_up), min(max_w, b[2] + dx), min(max_h, b[3] + dy_down)]

# ─── 1.5 IMAGE ENHANCEMENT (CLAHE) ───
def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# ─── 2. GEOMETRIC FEATURE EXTRACTION (FOR ML MODEL) ───
def extract_geometric_features(motos, persons):
    features = []
    for moto in motos:
        mx1, my1, mx2, my2 = moto
        moto_w, moto_h = mx2 - mx1, my2 - my1
        moto_area = moto_w * moto_h

        overlaps, person_centers_x, person_centers_y, person_areas = [], [], [], []

        for person in persons:
            px1, py1, px2, py2 = person
            x_left, y_top = max(mx1, px1), max(my1, py1)
            x_right, y_bottom = min(mx2, px2), min(my2, py2)
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                person_area = (px2 - px1) * (py2 - py1)
                if person_area > 0:
                    overlap_ratio = intersection / person_area
                    if overlap_ratio > 0.01:
                        overlaps.append(overlap_ratio)
                        person_centers_x.append((px1 + px2) / 2)
                        person_centers_y.append((py1 + py2) / 2)
                        person_areas.append(person_area)

        if len(overlaps) == 0: continue

        spread_x = (max(person_centers_x) - min(person_centers_x)) if len(person_centers_x) > 1 else 0
        spread_y = (max(person_centers_y) - min(person_centers_y)) if len(person_centers_y) > 1 else 0

        features.append({
            "n_overlapping": len(overlaps),
            "max_overlap_ratio": max(overlaps),
            "mean_overlap_ratio": np.mean(overlaps),
            "horizontal_spread_ratio": spread_x / moto_w if moto_w > 0 else 0,
            "vertical_spread_ratio": spread_y / moto_h if moto_h > 0 else 0,
            "moto_area": moto_area,
            "mean_person_area": np.mean(person_areas),
            "moto_box": moto,
        })
    return features


# ─── 2.5 THE UNIFIED RULE ENGINE (v2) ───
CLASSIFIER_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def rule_engine_unified(yolo_results, frame, helmet_model, seatbelt_model,
                         helmet_conf=0.60, seatbelt_conf=0.60, triple_riding_conf=0.60,
                         overlap_thresh=0.15, moto_expand_frac=0.25):
    violations = []
    if not yolo_results or yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
        return violations

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w

    boxes = yolo_results[0].boxes.xyxy.cpu()
    cls_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    confs = yolo_results[0].boxes.conf.cpu()
    cls_names = yolo_results[0].names

    motos_raw = [boxes[i] for i in range(len(boxes)) if cls_names[cls_ids[i]].lower() in ('motorcycle', 'motorbike', 'bike', 'bicycle')]
    persons_raw = [(boxes[i], confs[i].item(), cls_names[cls_ids[i]].lower()) for i in range(len(boxes)) if cls_names[cls_ids[i]].lower() in ('person', 'rider')]
    cars_raw = [boxes[i] for i in range(len(boxes)) if cls_names[cls_ids[i]].lower() in ('car', 'truck', 'bus')]

    # ─── 1. STANDARD NMS ───
    def run_nms(box_list, iou_thresh=0.45):
        if not box_list: return []
        b_tensor = torch.stack(box_list)
        s_tensor = torch.ones(len(box_list)) 
        keep_idx = torchvision.ops.nms(b_tensor, s_tensor, iou_thresh)
        return [box_list[i].numpy() for i in keep_idx]

    motos = run_nms(motos_raw, 0.45)
    cars = run_nms(cars_raw, 0.45)

    persons = []
    if persons_raw:
        p_boxes = torch.stack([p[0] for p in persons_raw])
        p_scores = torch.tensor([p[1] for p in persons_raw])
        keep_idx = torchvision.ops.nms(p_boxes, p_scores, 0.45)
        for idx in keep_idx:
            persons.append((p_boxes[idx].numpy(), p_scores[idx].item(), persons_raw[idx][2]))

    # ─── 2. ROBUST CENTER-POINT ASSIGNMENT ───
    moto_to_riders = {m_idx: [] for m_idx in range(len(motos))}
    unassigned_riders = []

    for person, conf, cls_name in persons:
        px1, py1, px2, py2 = person
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        
        assigned = False
        for m_idx, moto in enumerate(motos):
            moto_expanded = _expand_box(moto, moto_expand_frac, frame_w, frame_h)
            # Assign if rider's center point falls physically inside the expanded motorcycle box
            if (moto_expanded[0] <= pcx <= moto_expanded[2] and moto_expanded[1] <= pcy <= moto_expanded[3]):
                moto_to_riders[m_idx].append(person)
                assigned = True
                break
        
        # Fallback Intersection Check
        if not assigned:
            best_moto = -1
            best_overlap = 0
            for m_idx, moto in enumerate(motos):
                moto_expanded = _expand_box(moto, moto_expand_frac, frame_w, frame_h)
                inter = _intersection_area(person, moto_expanded)
                person_area = _bbox_area(person)
                overlap = inter / person_area if person_area > 0 else 0
                if overlap > overlap_thresh and overlap > best_overlap:
                    best_overlap = overlap
                    best_moto = m_idx
            if best_moto != -1:
                moto_to_riders[best_moto].append(person)
                assigned = True

        if not assigned and cls_name == 'rider':
            unassigned_riders.append(person)

    # ─── 3. TRIPLE RIDING (Rule & ML) ───
    for m_idx, moto in enumerate(motos):
        riders_on_this_moto = moto_to_riders[m_idx]
        global triple_riding_clf
        
        if len(riders_on_this_moto) >= 2:
            cand_feats = extract_geometric_features([moto], riders_on_this_moto)
            if cand_feats:
                cand = cand_feats[0]
                feature_cols = ["n_overlapping", "max_overlap_ratio", "mean_overlap_ratio",
                                "horizontal_spread_ratio", "vertical_spread_ratio",
                                "moto_area", "mean_person_area"]
                
                flagged_by_rule = (
                    cand["n_overlapping"] >= 3
                )
                
                flagged_by_model = False
                learned_prob = 0.0
                if triple_riding_clf is not None:
                    feat_vec = np.array([[cand[c] for c in feature_cols]])
                    learned_prob = triple_riding_clf.predict_proba(feat_vec)[0, 1]
                    flagged_by_model = learned_prob >= triple_riding_conf
                    
                if flagged_by_rule or flagged_by_model:
                    confidence = max(cand["mean_overlap_ratio"], learned_prob)
                    flag_source = "Both" if (flagged_by_rule and flagged_by_model) else ("ML Model" if flagged_by_model else "Rule")
                    violations.append({
                        "type": f"Triple Riding ({flag_source})",
                        "confidence": float(confidence),
                        "bbox": [int(x) for x in moto],
                        "vehicle_bbox": [int(x) for x in moto],
                        "color": (255, 0, 255),
                    })

    # ─── 4. UNIFIED NO HELMET CHECK ───
    all_riders_to_check = []
    for riders in moto_to_riders.values():
        all_riders_to_check.extend(riders)
    all_riders_to_check.extend(unassigned_riders)

    for rider in all_riders_to_check:
        rx1, ry1, rx2, ry2 = [int(x) for x in rider]
        box_w, box_h = rx2 - rx1, ry2 - ry1
        
        head_h = int(box_h * 0.40)
        head_y1 = max(0, ry1 - int(box_h * 0.10))
        head_y2 = ry1 + head_h
        crop_x1 = max(0, rx1 - int(box_w * 0.05))
        crop_x2 = min(frame_w, rx2 + int(box_w * 0.05))
        if crop_x2 <= crop_x1: crop_x1, crop_x2 = rx1, rx2

        head_crop = frame[head_y1:head_y2, crop_x1:crop_x2]
        if head_crop.size == 0: continue

        shift_y = max(2, int(box_h * 0.05))
        crops = [head_crop]
        if head_y1 >= shift_y: crops.append(frame[head_y1 - shift_y : head_y2 - shift_y, crop_x1:crop_x2])
        if head_y2 <= frame_h - shift_y: crops.append(frame[head_y1 + shift_y : head_y2 + shift_y, crop_x1:crop_x2])
            
        probs = []
        for c in crops:
            if c.size == 0: continue
            pil_img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            tensor_img = CLASSIFIER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = helmet_model(tensor_img)
                probs.append(torch.softmax(outputs, dim=1)[0].cpu().numpy())
        
        if not probs: continue
        prob_without_helmet = float(max([p[1] for p in probs]))

        if prob_without_helmet >= helmet_conf:
            violations.append({
                'type': 'No Helmet', 'confidence': prob_without_helmet,
                'bbox': [rx1, ry1, rx2, ry2],
                'vehicle_bbox': [rx1, ry1, rx2, ry2],
                'color': (0, 0, 255),
            })

    # ─── 5. NO SEATBELT CHECK ───
    min_car_area = frame_area * 0.01
    for car in cars:
        cx1, cy1, cx2, cy2 = [int(x) for x in car]
        car_area = (cx2 - cx1) * (cy2 - cy1)
        if car_area < min_car_area: continue

        # YOLO bounding boxes are often loose (including the wall/roof).
        # We take multiple crops focusing on the right-hand-drive driver area (left side of car box)
        # to guarantee the Seatbelt Classifier gets a perfectly aligned view.
        w, h = cx2 - cx1, cy2 - cy1
        crops = []
        
        # 1. Broad Windshield Crop
        crops.append((frame[cy1 : cy1 + int(h * 0.45), cx1:cx2], cy1 + int(h * 0.45)))
        
        # 2. Driver-Half Crop (Left side of image for right-hand drive)
        crops.append((frame[cy1 + int(h * 0.15) : cy1 + int(h * 0.60), cx1 + int(w * 0.05) : cx1 + int(w * 0.55)], cy1 + int(h * 0.60)))
        
        # 3. Tight Driver Chest Zoom
        crops.append((frame[cy1 + int(h * 0.25) : cy1 + int(h * 0.55), cx1 + int(w * 0.15) : cx1 + int(w * 0.45)], cy1 + int(h * 0.55)))

        max_prob = 0.0
        best_mid_y = cy1 + int(h * 0.45)
        
        for crop_img, mid_y_ref in crops:
            if crop_img.shape[0] < 10 or crop_img.shape[1] < 10: continue
                
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            tensor_img = CLASSIFIER_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = seatbelt_model(tensor_img)
                probs = torch.softmax(outputs, dim=1)[0]
                
            prob_no_seatbelt = float(probs[0].item())
            if prob_no_seatbelt > max_prob:
                max_prob = prob_no_seatbelt
                best_mid_y = mid_y_ref
                
        with open("debug_run.log", "a", encoding="utf-8") as f_log:
            f_log.write(f"  [Seatbelt Debug] car bbox {cx1},{cy1},{cx2},{cy2} -> max_prob_no_seatbelt={max_prob:.4f}\n")
                
        if max_prob >= seatbelt_conf:
            violations.append({
                'type': 'No Seatbelt', 'confidence': max_prob,
                'bbox': [cx1, cy1, cx2, best_mid_y],
                'vehicle_bbox': [cx1, cy1, cx2, cy2],
                'color': (0, 165, 255),
            })

    return violations

# ─── 3. LOAD MODELS ───
YOLO_PATH = r"C:\Users\akshi\Downloads\best.pt"
HELMET_CKPT = r"C:\Users\akshi\Downloads\grid\demo_app\best_helmet_efficientnet_b3.pt"
SEATBELT_CKPT = r"C:\Users\akshi\Downloads\grid\demo_app\best_seatbelt_mobilenetv3_large.pt"
TRIPLE_RIDING_PKL = r"C:\Users\akshi\Downloads\triple_riding_learned_classifier.pkl"

detector = None
helmet_model = None
seatbelt_model = None
triple_riding_clf = None
paddle_ocr_model = None

def load_models_lazy():
    global detector, helmet_model, seatbelt_model, triple_riding_clf, paddle_ocr_model
    
    if paddle_ocr_model is None and PaddleOCR is not None:
        try:
            paddle_ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("✅ Successfully loaded PaddleOCR!")
        except Exception as e:
            print(f"Error loading PaddleOCR: {e}")
    
    if triple_riding_clf is None:
        try:
            if os.path.exists(TRIPLE_RIDING_PKL):
                triple_riding_clf = joblib.load(TRIPLE_RIDING_PKL)
                print("✅ Successfully loaded Ensemble Triple Riding ML model!")
            else:
                print(f"⚠️ Could not find ML model at {TRIPLE_RIDING_PKL}. Falling back to Rule Engine only.")
        except Exception as e:
            print(f"Error loading Triple Riding ML model: {e}")
    if detector is None:
        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_PATH):
                detector = YOLO(YOLO_PATH)
            else:
                detector = YOLO("yolov8n.pt")
                print("⚠️ Using default coco yolov8n.pt")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            
    if helmet_model is None:
        try:
            helmet_model = build_efficientnet_b3(len(HELMET_CLASSES))
            if os.path.exists(HELMET_CKPT):
                helmet_model.load_state_dict(torch.load(HELMET_CKPT, map_location=DEVICE))
            helmet_model.to(DEVICE)
            helmet_model.eval()
        except Exception as e:
            print(f"Error loading Helmet model: {e}")
            
    if seatbelt_model is None:
        try:
            seatbelt_model = build_mobilenet_large(len(SEATBELT_CLASSES))
            if os.path.exists(SEATBELT_CKPT):
                seatbelt_model.load_state_dict(torch.load(SEATBELT_CKPT, map_location=DEVICE))
            seatbelt_model.to(DEVICE)
            seatbelt_model.eval()
        except Exception as e:
            print(f"Error loading Seatbelt model: {e}")

# ─── 4. GRADIO INFERENCE PIPELINE FUNCTION ───
def process_image(input_image, use_clahe, helmet_conf, seatbelt_conf, triple_riding_conf, overlap_thresh, moto_expand):
    if input_image is None:
        return None, "Please upload an image first."
    
    load_models_lazy()
    
    frame = np.array(input_image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if use_clahe:
        frame_bgr = apply_clahe_bgr(frame_bgr)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        results = detector.predict(source=frame_bgr, conf=0.20, imgsz=1024, augment=True, verbose=False, device=DEVICE, agnostic_nms=True)
    except Exception as e:
        return None, f"Inference Error in YOLO model: {e}"
        
    try:
        import time
        with open("debug_run.log", "a", encoding="utf-8") as f_log:
            f_log.write(f"\n--- Run at {time.ctime()} ---\n")
            f_log.write(f"Image shape: {frame.shape}\n")
            if results and len(results) > 0 and results[0].boxes is not None:
                f_log.write(f"YOLO detected {len(results[0].boxes)} boxes:\n")
                names = results[0].names
                for i in range(len(results[0].boxes)):
                    c = int(results[0].boxes.cls[i].item())
                    conf = float(results[0].boxes.conf[i].item())
                    box = results[0].boxes.xyxy[i].cpu().numpy().tolist()
                    f_log.write(f"  - Box {i}: {names[c]} (conf={conf:.2f}) bbox={box}\n")
            else:
                f_log.write("YOLO detected no boxes.\n")
    except Exception as log_err:
        pass
        
    annotated_img = frame_bgr.copy()
    if results and len(results) > 0 and results[0].boxes is not None:
        import torchvision
        boxes = results[0].boxes.xyxy.cpu()
        scores = results[0].boxes.conf.cpu()
        keep_idx = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
        
        names = results[0].names
        for i in keep_idx:
            if scores[i] < 0.40:
                continue
            box = boxes[i]
            x1, y1, x2, y2 = [int(x) for x in box]
            cls_id = int(results[0].boxes.cls[i].item())
            class_name = names[cls_id].lower()
            
            # Color palette (BGR) - Darker/Deeper colors
            if class_name in ['car']: color = (200, 0, 0) # Deep Blue
            elif class_name in ['bus', 'truck']: color = (0, 0, 150) # Deep Red
            elif class_name in ['motorcycle', 'bicycle']: color = (0, 100, 200) # Deep Orange
            elif class_name in ['person', 'rider']: color = (0, 100, 0) # Dark Green
            elif class_name in ['autorickshaw']: color = (150, 0, 150) # Deep Purple
            else: color = (50, 50, 50) # Dark Grey default
            
            # Draw thicker colored wireframe
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
            
            # Draw colored background box for text with white text for contrast
            (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated_img, (x1, y1), (x1 + tw + 6, y1 + th + 8), color, -1)
            cv2.putText(annotated_img, class_name, (x1 + 3, y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    try:
        violations = rule_engine_unified(
            results, frame_bgr, helmet_model, seatbelt_model,
            helmet_conf=helmet_conf, seatbelt_conf=seatbelt_conf, triple_riding_conf=triple_riding_conf,
            overlap_thresh=overlap_thresh, moto_expand_frac=moto_expand
        )
        
        # CONDITIONAL ANPR & EVIDENCE PACKET GENERATION
        frame_h, frame_w = frame_bgr.shape[:2]
        current_timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        
        for v in violations:
            v['timestamp'] = current_timestamp
            v['license_plate'] = "Not Detected"
            v['ocr_confidence'] = 0.0
            
            # Use the vehicle's bounding box for OCR, not the violation box (which might just be the rider's face)
            ocr_box = v.get('vehicle_bbox', v['bbox'])
            x1, y1, x2, y2 = [int(coord) for coord in ocr_box]
            
            # Expand the crop slightly to ensure the license plate is caught (it might be near the edge)
            pad = int((x2 - x1) * 0.1)
            cx1 = max(0, x1 - pad)
            cx2 = min(frame_w, x2 + pad)
            cy1 = max(0, y1 - pad)
            cy2 = min(frame_h, y2 + pad)
            
            crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
            
            if crop_bgr.size > 0 and paddle_ocr_model is not None:
                try:
                    ocr_result = paddle_ocr_model.ocr(crop_bgr, cls=True)
                    if ocr_result and ocr_result[0]:
                        best_text = ""
                        best_conf = 0.0
                        for line in ocr_result[0]:
                            text, conf = line[1]
                            if conf > best_conf and len(text.strip()) >= 4:
                                best_text = text.strip()
                                best_conf = float(conf)
                        
                        if best_text:
                            v['license_plate'] = best_text
                            v['ocr_confidence'] = best_conf
                except Exception as e:
                    pass

        with open("debug_run.log", "a", encoding="utf-8") as f_log:
            f_log.write(f"Rule Engine returned {len(violations)} violations:\n")
            for v in violations:
                f_log.write(f"  - {v['type']} (conf={v['confidence']:.2f}) plate={v['license_plate']} bbox={v['bbox']}\n")
    except Exception as e:
        try:
            with open("debug_run.log", "a", encoding="utf-8") as f_log:
                f_log.write(f"Rule Engine crashed: {e}\n")
        except:
            pass
        return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), f"YOLO succeeded, but Rule Engine crashed: {e}"
        
    for v in violations:
        x1, y1, x2, y2 = v['bbox']
        color = v['color']
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 4)
        lbl = f"🚨 {v['type']} ({v['confidence']:.2f})"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_img, (x1, max(y1 - th - 10, 0)), (x1 + tw + 4, max(y1, th + 4)), color, -1)
        cv2.putText(annotated_img, lbl, (x1 + 2, max(y1 - 5, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    output_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    if not violations:
        html_report = """
        <div style="background-color: #d4edda; border-left: 6px solid #28a745; padding: 15px; border-radius: 4px;">
            <h3 style="color: #155724; margin: 0;">✅ Safe Road Conditions</h3>
            <p style="color: #155724; margin: 5px 0 0 0;">No traffic violations detected in this frame.</p>
        </div>
        """
    else:
        html_report = f"""
        <div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
            <h3 style="color: #856404; margin: 0;">🚨 {len(violations)} Violation(s) Detected & Evidence Captured</h3>
        </div>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-family: sans-serif;">
            <thead>
                <tr style="background-color: #f2f2f2; text-align: left; border-bottom: 2px solid #ddd;">
                    <th style="padding: 10px;">Violation Type</th>
                    <th style="padding: 10px;">Confidence</th>
                    <th style="padding: 10px;">License Plate (OCR)</th>
                </tr>
            </thead>
            <tbody>
        """
        for v in violations:
            badge_color = "#E74C3C" if v['type'] == 'No Helmet' else ("#ffc107" if v['type'] == 'No Seatbelt' else "#9b59b6")
            text_color = "#fff"
            plate_display = f"<strong>{v['license_plate']}</strong><br><span style='font-size:0.8em; color:#666;'>Conf: {v['ocr_confidence']*100:.1f}%</span>" if v['license_plate'] != "Not Detected" else "<span style='color:#999;'>Not Detected</span>"
            
            html_report += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 10px; font-weight: bold;"><span style="background-color: {badge_color}; color: {text_color}; padding: 3px 8px; border-radius: 4px; font-size: 0.85em;">{v['type']}</span></td>
                    <td style="padding: 10px;">{v['confidence']*100:.1f}%</td>
                    <td style="padding: 10px;">{plate_display}</td>
                </tr>
            """
        html_report += "</tbody></table>"
        
    return output_rgb, html_report

# ─── 5. THE GRAPHICAL INTERFACE (GRADIO) ───
css = """
body { background-color: #0d1117; color: #c9d1d9; margin: 0 !important; padding: 0 !important; }
.gradio-container { max-width: 100vw !important; margin: 0 !important; padding: 0 !important; width: 100vw !important; overflow-x: hidden; border: none !important; }
.contain { padding: 0 !important; margin: 0 !important; max-width: 100vw !important; }
.wrap { padding: 0 !important; margin: 0 !important; max-width: 100vw !important; }
h1 { text-align: center; font-family: 'Inter', sans-serif; font-weight: 800; margin-bottom: 5px; }
.desc { text-align: center; font-size: 1.1em; color: #8b949e; margin-bottom: 25px; }

/* Hide Gradio Progress and Queue text */
.progress-text { display: none !important; }
.meta-text { display: none !important; }
.eta-bar { display: none !important; }
.progress-level { display: none !important; }
"""

with gr.Blocks(css=css) as demo:
    with gr.Tabs():
        with gr.Tab("🏠 Home"):
            import base64
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base_dir, "indian-city-scene.jpg"), "rb") as f1:
                bg_b64 = base64.b64encode(f1.read()).decode('utf-8')
            with open(os.path.join(base_dir, "architecture_grid_lock.jpeg"), "rb") as f2:
                arch_b64 = base64.b64encode(f2.read()).decode('utf-8')

            home_html = f"""
            <div style="width: 100vw; margin: -20px; padding: 0; box-sizing: border-box; overflow-x: hidden;">
              <!-- Background Image Hero Section -->
              <div style="background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.8)), url('data:image/jpeg;base64,{bg_b64}'); background-size: cover; background-position: center; min-height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; padding: 40px 20px; box-sizing: border-box;">
                <div style="text-align: center; margin-bottom: 3rem; width: 100%;">
                  <h1 style="color: #FBBF24; font-size: 6em; margin-bottom: 0px; text-shadow: 4px 4px 8px rgba(0,0,0,0.8);">GRIDLOCK</h1>
                  <p style="color: #FFFFFF; font-size: 1.5em; margin-top: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);">Automated Traffic Violation Detection</p>
                </div>
                
                <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                    <!-- Team Details -->
                    <div style="background: rgba(13, 17, 23, 0.85); backdrop-filter: blur(10px); padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); width: 100%; max-width: 600px;">
                      <h2 style="color: #FBBF24; text-align: center; margin-bottom: 1rem; font-size: 2em;">Team Elite Hackers</h2>
                      <ul style="list-style-type: none; padding: 0; text-align: center;">
                        <li style="color: #FFFFFF; font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Akshitha Reddy</li>
                        <li style="color: #FFFFFF; font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Shreya</li>
                        <li style="color: #FFFFFF; font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Pradeep</li>
                        <li style="color: #FFFFFF; font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">Sandeep</li>
                      </ul>
                      <hr style="border: 1px solid rgba(255,255,255,0.2); margin: 20px 0;">
                      <p style="color: #8b949e; text-align: center; margin-top: 1rem; font-size: 1.2em; text-transform: uppercase;">Vasavi College of Engineering</p>
                    </div>
                </div>
              </div>
              
              <!-- Architecture Section (Outside Background) -->
              <div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 40px 20px; box-sizing: border-box; background-color: transparent;">
                  <!-- Architecture Details -->
                  <div style="background: rgba(13, 17, 23, 0.85); padding: 2rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); width: 100%; max-width: 1200px;">
                    <details style="cursor: pointer;">
                        <summary style="color: #FBBF24; font-size: 1.8em; font-weight: bold; text-align: center; list-style: none; outline: none;">
                            📐 Click to View Pipeline Architecture
                        </summary>
                        <div style="margin-top: 20px; text-align: center;">
                            <img src="data:image/jpeg;base64,{arch_b64}" alt="Architecture Diagram" style="max-width: 100%; border-radius: 10px; border: 2px solid #FBBF24; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                        </div>
                    </details>
                  </div>
              </div>
            </div>
            """
            gr.HTML(home_html)

        with gr.Tab("🚦 Traffic Analysis"):
            gr.Markdown("# 🚦 GRIDLOCK: Traffic Violation AI Pipeline Demo")
            gr.Markdown(
                "<p class='desc'>Upload road images to detect motorcycle Helmet violations, Seatbelt violations, and Triple Riding geometry on the fly.</p>"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="pil", label="📸 Upload Traffic Image")
                    
                    with gr.Accordion("⚙️ Advanced Pipeline Parameters", open=False):
                        use_clahe_cb = gr.Checkbox(label="Apply CLAHE (Contrast Enhancement)", value=False)
                        h_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.60, step=0.05, label="Helmet Classifier Conf")
                        s_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Seatbelt Classifier Conf")
                        t_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.60, step=0.05, label="Triple Riding ML Conf")
                        over_th = gr.Slider(minimum=0.01, maximum=0.8, value=0.05, step=0.01, label="Rider-Moto Overlap threshold")
                        moto_exp = gr.Slider(minimum=0.0, maximum=0.5, value=0.25, step=0.05, label="Motorcycle Box Expansion Frac")
                        
                    submit_btn = gr.Button("⚡ Analyze Traffic Frame", variant="primary")
                    
                with gr.Column(scale=1.2):
                    output_img = gr.Image(type="numpy", label="👁️ AI Annotated Output")
                    output_html = gr.HTML(label="📋 Violation Report Summary")

            submit_btn.click(
                fn=process_image,
                inputs=[input_img, use_clahe_cb, h_conf, s_conf, t_conf, over_th, moto_exp],
                outputs=[output_img, output_html]
            )
            
            gr.Examples(
                examples=[],
                inputs=input_img,
                label="Select a Sample Image"
            )

demo.queue().launch()