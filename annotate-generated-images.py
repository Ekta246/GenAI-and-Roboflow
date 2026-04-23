"""
Auto-Annotation Script for Generated Fire/Smoke Images
Supports two backends:
  - "sam3"  : Roboflow SAM3 workflow (zero-shot, no custom model needed)
  - "yolo"  : Local YOLO model inference

Workflow:
1. Generate images with generate-images-grok.py
2. Run this script to auto-annotate them
3. Review and filter annotations if needed
"""

import os
import cv2
import time
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ===================== CONFIGURATION =====================
CONFIG = {
    # Backend: "sam3" or "yolo"
    "backend": "sam3",

    # --- SAM3 settings (used when backend == "sam3") ---
    "sam3_api_url": "https://serverless.roboflow.com",
    "sam3_api_key": os.getenv("ROBOFLOW_API_KEY", ""),
    "sam3_workspace": "intellisee",
    "sam3_workflow_id": "sam3-with-prompts",
    "sam3_prompts": ["fire", "smoke", "car", "truck", "bike", "motorcycle", "bus", "vehicle"],

    # --- YOLO settings (used when backend == "yolo") ---
    "model_path": "/Users/ekta/Downloads/yolo11-d-fire-dataset.pt",
    "iou_threshold": 0.5,
    "class_names": {0: "smoke", 1: "fire"},

    # --- Common settings ---
    "input_images_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/sf-images/images",
    "output_annotations_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/sf-images/annotations",

    "conf_threshold": 0.25,
    "min_box_area": 100,
    "save_only_with_detections": True,

    "save_visualized_images": True,
    "visualization_output_dir": None,  # None → input_images_dir/visualized
}


# ===================== FUNCTIONS =====================

def draw_bounding_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    vis_image = image.copy()
    
    # Color mapping for classes
    colors = {
        "fire": (0, 0, 255),    # Red for fire
        "smoke":(150, 165, 250), # Orange for smoke
    }
    
    for det in detections:
        xmin, ymin, xmax, ymax = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        class_name = det["name"]
        confidence = det["confidence"]
        
        # Get color for this class
        color = colors.get(class_name, (0, 255, 0))  # Default green for unknown
        
        # Draw bounding box
        cv2.rectangle(vis_image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            vis_image,
            (xmin, ymin - text_height - baseline - 5),
            (xmin + text_width, ymin),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis_image,
            label,
            (xmin, ymin - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1
        )
    
    return vis_image


def save_pascal_voc_xml(image_path: str, image_shape: tuple, detections: list, output_path: str):
    """Save detections in Pascal VOC XML format."""
    annotation = ET.Element("annotation")
    
    # Folder and filename
    ET.SubElement(annotation, "folder").text = os.path.dirname(image_path)
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = image_path
    
    # Source
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Grok Generated Fire Dataset"
    
    # Image size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_shape[1])
    ET.SubElement(size, "height").text = str(image_shape[0])
    ET.SubElement(size, "depth").text = str(image_shape[2] if len(image_shape) > 2 else 3)
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    # Objects
    for det in detections:
        obj = ET.SubElement(annotation, "object")
        # Class name is always "fire" or "smoke" (never "0" or "1")
        class_name = det["name"]
        # Validate class name before writing to XML
        if class_name not in ["fire", "smoke"] and not class_name.startswith("class_"):
            print(f"⚠️  Warning: Unexpected class name '{class_name}' in XML, expected 'fire' or 'smoke'")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(det["xmin"]))
        ET.SubElement(bbox, "ymin").text = str(int(det["ymin"]))
        ET.SubElement(bbox, "xmax").text = str(int(det["xmax"]))
        ET.SubElement(bbox, "ymax").text = str(int(det["ymax"]))
    
    # Write XML
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def detect_with_sam3(image_path: str) -> list:
    """Run SAM3 workflow on a single image and return detections."""
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url=CONFIG["sam3_api_url"],
        api_key=CONFIG["sam3_api_key"],
    )

    sam_data = client.run_workflow(
        workspace_name=CONFIG["sam3_workspace"],
        workflow_id=CONFIG["sam3_workflow_id"],
        images={"image": image_path},
        parameters={"prompts": CONFIG["sam3_prompts"]},
        use_cache=True,
    )

    detections = []
    try:
        predictions = sam_data[0]["sam"]["predictions"]
    except (IndexError, KeyError, TypeError):
        predictions = []

    vehicle_labels = {"car", "truck", "bike", "motorcycle", "bus", "vehicle", "van", "suv"}

    for pred in predictions:
        class_name = pred.get("class", "unknown").lower()
        if class_name in vehicle_labels:
            class_name = "vehicle"
        elif class_name not in ("fire", "smoke"):
            continue
        conf = pred.get("confidence", 0)
        if conf < CONFIG["conf_threshold"]:
            continue

        cx = pred.get("x", 0)
        cy = pred.get("y", 0)
        w = pred.get("width", 0)
        h = pred.get("height", 0)
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        if w * h < CONFIG["min_box_area"]:
            continue

        detections.append({
            "name": class_name,
            "confidence": conf,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return detections


def detect_with_yolo(image: np.ndarray) -> list:
    """Run YOLO model on a single image and return detections."""
    from ultralytics import YOLO

    if not hasattr(detect_with_yolo, "_model"):
        detect_with_yolo._model = YOLO(CONFIG["model_path"])
    model = detect_with_yolo._model
    class_names = CONFIG["class_names"]

    results = model.predict(
        source=image,
        conf=CONFIG["conf_threshold"],
        iou=CONFIG["iou_threshold"],
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i])
            cls = int(boxes.cls[i])
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)
            if area < CONFIG["min_box_area"]:
                continue
            class_name = class_names.get(cls, f"class_{cls}")
            detections.append({
                "name": class_name,
                "confidence": conf,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })
    return detections


def process_images():
    """Process all generated images and create annotations."""

    output_dir = Path(CONFIG["output_annotations_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(CONFIG["input_images_dir"])
    image_files = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    )

    backend = CONFIG["backend"]
    print(f"\n  Backend: {backend.upper()}")
    if backend == "sam3":
        print(f"  Workflow: {CONFIG['sam3_workspace']}/{CONFIG['sam3_workflow_id']}")
        print(f"  Prompts: {CONFIG['sam3_prompts']}")
    else:
        print(f"  Model: {CONFIG['model_path']}")

    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images: {len(image_files)}")
    print(f"  Confidence threshold: {CONFIG['conf_threshold']}")
    print(f"{'=' * 60}")

    vis_output_dir = None
    if CONFIG["save_visualized_images"]:
        vis_output_dir = Path(CONFIG["visualization_output_dir"]) if CONFIG["visualization_output_dir"] else input_dir / "visualized"
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Visualizations: {vis_output_dir}")

    total_processed = 0
    total_with_detections = 0
    total_detections = 0
    class_counts = {"fire": 0, "smoke": 0}

    for idx, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not read: {img_path.name}")
            continue

        t0 = time.time()
        if backend == "sam3":
            detections = detect_with_sam3(str(img_path))
        else:
            detections = detect_with_yolo(image)
        elapsed = time.time() - t0

        if not detections and CONFIG["save_only_with_detections"]:
            print(f"[{idx+1}/{len(image_files)}] {img_path.name} -> 0 detections ({elapsed:.1f}s) - skipped")
            continue

        xml_path = output_dir / (img_path.stem + ".xml")
        save_pascal_voc_xml(str(img_path), image.shape, detections, str(xml_path))

        total_processed += 1
        if detections:
            total_with_detections += 1
            total_detections += len(detections)
            for d in detections:
                class_counts[d["name"]] = class_counts.get(d["name"], 0) + 1

            det_summary = ", ".join([f"{d['name']} {d['confidence']:.2f}" for d in detections])
            print(f"[{idx+1}/{len(image_files)}] {img_path.name} -> {len(detections)} det ({elapsed:.1f}s): {det_summary}")

            if vis_output_dir:
                vis_image = draw_bounding_boxes(image, detections)
                vis_path = vis_output_dir / (img_path.stem + "_annotated" + img_path.suffix)
                cv2.imwrite(str(vis_path), vis_image)
        else:
            print(f"[{idx+1}/{len(image_files)}] {img_path.name} -> 0 detections ({elapsed:.1f}s)")

    print(f"\n{'=' * 60}")
    print(f"  Annotation Complete!")
    print(f"{'=' * 60}")
    rate = (total_with_detections / len(image_files) * 100) if image_files else 0
    print(f"  Images processed:       {total_processed}")
    print(f"  Images with detections: {total_with_detections}")
    print(f"  Total detections:       {total_detections}")
    print(f"  Detection rate:         {rate:.1f}%")
    print(f"  Annotations saved to:   {output_dir}")
    if vis_output_dir:
        print(f"  Visualizations saved:   {vis_output_dir}")
    print(f"\n  Class breakdown:")
    for cls_name, count in class_counts.items():
        if count > 0:
            print(f"    {cls_name}: {count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║   Auto-Annotation for Fire/Smoke Images                   ║
    ║   Backend: {CONFIG['backend'].upper():<46s}║
    ╚════════════════════════════════════════════════════════════╝
    """)

    process_images()

