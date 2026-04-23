import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/Users/ekta/Downloads/yolo11-d-fire-dataset.pt"
IMAGE_DIR = "/Users/ekta/Downloads/fire/train/images"
OUTPUT_DIR = "/Users/ekta/Downloads/fire/train/outputs/annotations"
CONF_THRES = 0.25
IOU_THRES = 0.5
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)
class_names = model.names  # {id: name}

def save_voc_xml(image_path, detections, save_path):
    img = cv2.imread(image_path)
    h, w, c = img.shape

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(annotation, "segmented").text = "0"

    for cls_id, conf, xmin, ymin, xmax, ymax in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_names[int(cls_id)]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    tree = ET.ElementTree(annotation)
    tree.write(save_path)

# ----------- Batch Inference ------------
results = model.predict(
    source=IMAGE_DIR,
    conf=CONF_THRES,
    iou=IOU_THRES,
    save=False,
    stream=True
)

for result in results:
    image_path = result.path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    xml_path = os.path.join(OUTPUT_DIR, f"{image_name}.xml")

    detections = []

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            detections.append((cls, score, xmin, ymin, xmax, ymax))

    save_voc_xml(image_path, detections, xml_path)

print("✅ Inference complete. Pascal VOC annotations saved.")
