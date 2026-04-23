"""
VLM-Based Annotation Script for Fire/Smoke/Vehicle Images
Uses a Vision Language Model (OpenAI GPT-4o, Anthropic Claude, or Google Gemini)
to detect fire, smoke, and vehicles and output Pascal VOC XML annotations.

Classes annotated: fire, smoke, vehicle

Workflow:
1. Point INPUT_IMAGES_DIR at your image folder
2. Set your chosen VLM provider and API key
3. Run: python vlm_annotate_images.py
4. Pascal VOC XML files are written to OUTPUT_ANNOTATIONS_DIR
"""

import os
import base64
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# ===================== CONFIGURATION =====================
CONFIG = {
    # ---- VLM Provider -------------------------------------------------------
    # Choose one: "openai"  |  "anthropic"  |  "gemini"
    "vlm_provider": "openai",

    # API keys – can also be set via environment variables
    "openai_api_key":    os.getenv("OPENAI_API_KEY", ""),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "gemini_api_key":    os.getenv("GEMINI_API_KEY", ""),

    # Model names (change to a newer/cheaper model if you prefer)
    "openai_model":    "gpt-4o",
    "anthropic_model": "claude-opus-4-5",
    "gemini_model":    "gemini-2.0-flash",

    # ---- Paths --------------------------------------------------------------
    "input_images_dir":     "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/grok_fire_images/images",
    "output_annotations_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/grok_fire_images/vlm_annotations",

    # ---- Detection settings -------------------------------------------------
    # Classes the VLM should look for
    "target_classes": ["fire", "smoke", "vehicle"],

    # Skip saving XML if no objects found?
    "save_only_with_detections": True,

    # Retry a failed API call up to this many times
    "max_retries": 3,
    "retry_delay_seconds": 5,

    # Seconds to wait between images (avoid rate-limit bursts)
    "request_delay_seconds": 1.0,

    # ---- Visualization ------------------------------------------------------
    "save_visualized_images": True,
    "visualization_output_dir": None,   # None → <input_images_dir>/vlm_visualized
}

# ===================== PASCAL VOC XML =====================

def save_pascal_voc_xml(image_path: str, image_shape: tuple, detections: list, output_path: str):
    """Save detections as a Pascal VOC XML annotation file."""
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = str(Path(image_path).parent)
    ET.SubElement(annotation, "filename").text = Path(image_path).name
    ET.SubElement(annotation, "path").text = str(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "VLM Annotated Fire/Smoke/Vehicle Dataset"

    h, w = image_shape[:2]
    depth = image_shape[2] if len(image_shape) > 2 else 3
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det["name"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(det["xmin"]))
        ET.SubElement(bbox, "ymin").text = str(int(det["ymin"]))
        ET.SubElement(bbox, "xmax").text = str(int(det["xmax"]))
        ET.SubElement(bbox, "ymax").text = str(int(det["ymax"]))

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ===================== VISUALIZATION =====================

CLASS_COLORS = {
    "fire":    (0,   0,   255),   # red
    "smoke":   (150, 150, 150),   # gray
    "vehicle": (0,   200, 0),     # green
}

def draw_bounding_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["xmin"]), int(det["ymin"]), int(det["xmax"]), int(det["ymax"])
        color = CLASS_COLORS.get(det["name"], (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = det["name"]
        if "confidence" in det:
            label += f" {det['confidence']:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - bl - 5), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


# ===================== PROMPT BUILDER =====================

SYSTEM_PROMPT = """\
You are an expert computer vision annotator specialising in fire safety surveillance footage.
Your job is to locate every instance of FIRE, SMOKE, and VEHICLE in an image and return their
bounding boxes in pixel coordinates.

Rules:
- Only label objects that clearly belong to one of the three classes: fire, smoke, vehicle.
- Coordinates must be integers clamped to the image dimensions.
- xmin < xmax and ymin < ymax.
- Return ONLY valid JSON – no markdown fences, no extra text.

Response schema (array of objects, empty array if nothing found):
[
  {
    "name": "fire",          // one of: fire | smoke | vehicle
    "confidence": 0.95,      // your confidence 0.0–1.0
    "xmin": 120,
    "ymin": 80,
    "xmax": 340,
    "ymax": 300
  }
]
"""

USER_PROMPT = (
    "Detect all instances of FIRE, SMOKE, and VEHICLE in this image. "
    "Return bounding boxes as JSON following the schema above. "
    "Image dimensions: {width}x{height} pixels."
)


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {"jpg": "image/jpeg", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp"}.get(ext, "image/jpeg")


# ===================== VLM BACKENDS =====================

def call_openai(image_path: str, width: int, height: int) -> list:
    from openai import OpenAI

    client = OpenAI(api_key=CONFIG["openai_api_key"])
    b64 = encode_image_base64(image_path)
    mime = image_mime_type(image_path)

    response = client.chat.completions.create(
        model=CONFIG["openai_model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": USER_PROMPT.format(width=width, height=height)},
                ],
            },
        ],
        max_tokens=2048,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    return parse_vlm_json(raw, width, height)


def call_anthropic(image_path: str, width: int, height: int) -> list:
    import anthropic

    client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
    b64 = encode_image_base64(image_path)
    mime = image_mime_type(image_path)

    response = client.messages.create(
        model=CONFIG["anthropic_model"],
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    },
                    {"type": "text", "text": USER_PROMPT.format(width=width, height=height)},
                ],
            }
        ],
    )
    raw = response.content[0].text.strip()
    return parse_vlm_json(raw, width, height)


def call_gemini(image_path: str, width: int, height: int) -> list:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=CONFIG["gemini_api_key"])
    model = genai.GenerativeModel(
        model_name=CONFIG["gemini_model"],
        system_instruction=SYSTEM_PROMPT,
    )
    img = Image.open(image_path)
    response = model.generate_content(
        [USER_PROMPT.format(width=width, height=height), img],
        generation_config={"temperature": 0, "max_output_tokens": 2048},
    )
    raw = response.text.strip()
    return parse_vlm_json(raw, width, height)


# ===================== JSON PARSER =====================

def parse_vlm_json(raw: str, width: int, height: int) -> list:
    """Parse VLM JSON output into a list of detection dicts."""
    # Strip any accidental markdown code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"    [WARN] Could not parse JSON response: {exc}")
        print(f"    Raw response: {raw[:300]}")
        return []

    valid_classes = set(CONFIG["target_classes"])
    detections = []
    for item in data:
        name = str(item.get("name", "")).lower().strip()
        if name not in valid_classes:
            continue

        try:
            xmin = max(0, int(item["xmin"]))
            ymin = max(0, int(item["ymin"]))
            xmax = min(width,  int(item["xmax"]))
            ymax = min(height, int(item["ymax"]))
        except (KeyError, ValueError, TypeError):
            continue

        if xmin >= xmax or ymin >= ymax:
            continue

        detections.append({
            "name": name,
            "confidence": float(item.get("confidence", 1.0)),
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })
    return detections


# ===================== DISPATCH =====================

def annotate_image(image_path: str, width: int, height: int) -> list:
    """Call the configured VLM and return detections for one image."""
    provider = CONFIG["vlm_provider"].lower()
    for attempt in range(1, CONFIG["max_retries"] + 1):
        try:
            if provider == "openai":
                return call_openai(image_path, width, height)
            elif provider == "anthropic":
                return call_anthropic(image_path, width, height)
            elif provider == "gemini":
                return call_gemini(image_path, width, height)
            else:
                raise ValueError(f"Unknown VLM provider: {provider!r}. "
                                 "Choose 'openai', 'anthropic', or 'gemini'.")
        except Exception as exc:
            print(f"    [ERROR] Attempt {attempt}/{CONFIG['max_retries']} failed: {exc}")
            if attempt < CONFIG["max_retries"]:
                time.sleep(CONFIG["retry_delay_seconds"])
    return []


# ===================== MAIN =====================

def process_images():
    input_dir = Path(CONFIG["input_images_dir"])
    output_dir = Path(CONFIG["output_annotations_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if CONFIG["save_visualized_images"]:
        vis_dir = (
            Path(CONFIG["visualization_output_dir"])
            if CONFIG["visualization_output_dir"]
            else input_dir / "vlm_visualized"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(input_dir.glob("*.png"))
        + list(input_dir.glob("*.jpg"))
        + list(input_dir.glob("*.jpeg"))
    )

    print(f"\nVLM provider : {CONFIG['vlm_provider']}")
    print(f"Target classes: {CONFIG['target_classes']}")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Images found: {len(image_files)}")
    print("=" * 60)

    total_processed = 0
    total_with_detections = 0
    total_detections = 0
    class_counts = {c: 0 for c in CONFIG["target_classes"]}

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}", end="  ", flush=True)

        # Load image to get dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            print("SKIP – could not read")
            continue
        h, w = image.shape[:2]

        # Query VLM
        detections = annotate_image(str(img_path), w, h)
        total_processed += 1

        if detections:
            total_with_detections += 1
            total_detections += len(detections)
            for det in detections:
                class_counts[det["name"]] = class_counts.get(det["name"], 0) + 1
            summary = ", ".join(
                f"{d['name']}({d['confidence']:.2f})" for d in detections
            )
            print(f"→ {len(detections)} object(s): {summary}")
        else:
            print("→ no detections")

        # Save Pascal VOC XML
        if detections or not CONFIG["save_only_with_detections"]:
            xml_path = output_dir / (img_path.stem + ".xml")
            save_pascal_voc_xml(str(img_path), image.shape, detections, str(xml_path))

        # Save visualization
        if CONFIG["save_visualized_images"] and detections:
            vis_img = draw_bounding_boxes(image, detections)
            vis_path = vis_dir / (img_path.stem + "_vlm_annotated" + img_path.suffix)
            cv2.imwrite(str(vis_path), vis_img)

        # Rate-limit pause
        if idx < len(image_files):
            time.sleep(CONFIG["request_delay_seconds"])

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("VLM Annotation Complete!")
    print(f"{'='*60}")
    print(f"  Total images processed : {total_processed}")
    print(f"  Images with detections : {total_with_detections}")
    print(f"  Total detections       : {total_detections}")
    if total_processed:
        rate = total_with_detections / total_processed * 100
        print(f"  Detection rate         : {rate:.1f}%")
    print(f"\n  Breakdown by class:")
    for cls, cnt in class_counts.items():
        print(f"    {cls:10s}: {cnt}")
    print(f"\n  Annotations saved to   : {output_dir}")
    if CONFIG["save_visualized_images"] and total_with_detections > 0:
        print(f"  Visualizations saved to: {vis_dir}")
    print(f"{'='*60}")


# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   VLM Auto-Annotation: Fire / Smoke / Vehicle             ║
    ║   Output format: Pascal VOC XML (object detection)        ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Minimal key validation before hitting the API
    provider = CONFIG["vlm_provider"].lower()
    key_map = {
        "openai":    CONFIG["openai_api_key"],
        "anthropic": CONFIG["anthropic_api_key"],
        "gemini":    CONFIG["gemini_api_key"],
    }
    if not key_map.get(provider):
        print(f"ERROR: No API key found for provider '{provider}'.")
        print(f"  Set it in CONFIG or export the corresponding environment variable.")
        raise SystemExit(1)

    process_images()
