import os
import json
import time
import base64
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------
# CONFIGURATION
# ----------------------------s

# Roboflow SAM3 Workflow (via inference_sdk)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing ROBOFLOW_API_KEY. Set it in your environment or .env file.")

rf_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# OpenAI GPT-5.2 API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in your environment or .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Image folder
IMAGE_FOLDER = "/Users/ekta/Downloads/test-set-smoke-fire/test/images-optional"
OUTPUT_FOLDER = "/Users/ekta/Downloads/outputs-response-smoke-fire"
VIS_FOLDER = os.path.join(OUTPUT_FOLDER, "visualizations")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)

# Color palette for different classes (BGR)
CLASS_COLORS = {
    "person": (0, 255, 0),
    "vehicle": (255, 165, 0),
    "gun": (0, 0, 255),
    "person with gun": (0, 0, 200),
    "person on ground": (0, 200, 200),
    "person sitting": (200, 200, 0),
    "person bending": (200, 0, 200),
    "smoke": (128, 128, 128),
    "fire": (0, 69, 255),
    "spill": (255, 255, 0),
    "snow": (255, 200, 200),
}
DEFAULT_COLOR = (255, 255, 255)


def draw_detections(image_path: str, objects: list, save_path: str):
    """Draw bounding boxes and labels on the image and save it."""
    img = cv2.imread(image_path)
    if img is None:
        return

    for obj in objects:
        label = obj["label"]
        conf = obj["confidence"]
        cx, cy, w, h = obj["bbox"]
        if any(v is None for v in [cx, cy, w, h]):
            continue

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(save_path, img)


# ----------------------------
# HELPER FUNCTION: PROCESS SINGLE IMAGE
# ----------------------------

def process_image(image_path):
    # 1️⃣ Send image to Roboflow SAM3 workflow via inference_sdk
    start_sam = time.time()
    sam_data = rf_client.run_workflow(
        workspace_name="intellisee",
        workflow_id="sam3-with-prompts",
        images={"image": image_path},
        parameters={"prompts": ["person standing", "vehicle","handgun", "person with gun", "person on ground", "person sitting", "person bending","smoke","fire","spill","snow","person on knees","person kneeling on floor with hands down"]},
        use_cache=True
    )
    end_sam = time.time()

    sam_latency = end_sam - start_sam
    
    # 2️⃣ Extract context from SAM3 output
    # run_workflow returns a list; predictions are nested under sam_data[0]["sam"]["predictions"]
    objects = []
    try:
        predictions = sam_data[0]["sam"]["predictions"]
        for pred in predictions:
            objects.append({
                "label": pred.get("class", "unknown"),
                "confidence": pred.get("confidence", 0),
                "bbox": [
                    pred.get("x"),
                    pred.get("y"),
                    pred.get("width"),
                    pred.get("height")
                ]
            })
    except (IndexError, KeyError, TypeError) as e:
        print(f"  Warning: Could not parse SAM response: {e}")
    
    context_text = f"Detected objects:\n{json.dumps(objects, indent=2)}"

    # 2.5 Visualize bounding boxes on image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(VIS_FOLDER, f"{base_name}_vis.jpg")
    draw_detections(image_path, objects, vis_path)
    print(f"  Saved visualization: {vis_path}")

    # 3️⃣ Send full image to GPT-5.2 for visual scene understanding
    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext.lstrip("."), "image/jpeg")

    llm_prompt = """You are an AI security threat assessment system analyzing a security camera image.

ROLE: Look at this image carefully and classify the threat level for dashboard alerting.

THINGS TO LOOK FOR:
- Firearms, weapons (guns, knives, bats)
- Fire, smoke
- Person fallen on the ground, person injured
- Aggressive behavior, fights
- Spills, hazardous conditions
- Vehicles in unusual positions
- Any other safety concern visible in the scene

THREAT CLASSIFICATION RULES:

CRITICAL (Immediate Attention Required):
- Any firearm/weapon visible
- Active fire
- Person fallen/on ground (potential medical emergency or assault)
- Person holding or pointing a weapon

HIGH RISK:
- Smoke (potential fire hazard)
- Person on ground with others standing nearby (possible assault)
- Aggressive postures or confrontation
- Fire near flammable materials

MEDIUM RISK:
- Spill on floor (slip hazard)
- Unusual crowd gathering
- Unattended objects
- Smoke in outdoor area

LOW RISK:
- Normal pedestrian activity
- Person exercising
- Vehicle parked normally
- No visible threats

RESPONSE FORMAT (strict JSON only, no extra text):
{
  "threat_level": "CRITICAL | HIGH | MEDIUM | LOW",
  "requires_immediate_attention": true/false,
  "dashboard_tag": "short tag (e.g., WEAPON DETECTED, FIRE ALERT, PERSON DOWN, SPILL HAZARD, ALL CLEAR)",
  "objects_of_concern": [
    {
      "object": "what you see",
      "risk": "why it is a concern"
    }
  ],
  "scene_summary": "One sentence describing what is happening in the image", When you see person sitting or squatting or pwerson on knees, in the tag add the action of the person saying its not fallen but position relative to the ground and if its dangerous
  "recommended_action": "What security personnel should do"
}

IMPORTANT:
- Describe what you actually SEE in the image, not assumptions.
- Prioritize human safety above all else.
- When in doubt, escalate — better to over-alert than miss a threat.
- Respond ONLY with the JSON. No extra text."""

    start_llm = time.time()
    llm_response = openai_client.responses.create(
        model="gpt-5.2",
        input=[
            {"role": "user", "content": [
                {"type": "input_text", "text": llm_prompt},
                {"type": "input_image", "image_url": f"data:{mime};base64,{img_b64}"}
            ]}
        ]
    )
    end_llm = time.time()

    llm_latency = end_llm - start_llm
    llm_text = llm_response.output_text

    # Try to parse LLM response as structured JSON
    try:
        llm_parsed = json.loads(llm_text)
    except json.JSONDecodeError:
        llm_parsed = {"raw_response": llm_text}

    threat_level = llm_parsed.get("threat_level", "UNKNOWN")
    dashboard_tag = llm_parsed.get("dashboard_tag", "N/A")
    needs_attention = llm_parsed.get("requires_immediate_attention", False)
    print(f"  [{threat_level}] {dashboard_tag} | Immediate attention: {needs_attention}")

    # 4️⃣ Save result
    output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_context.json")

    with open(output_file, "w") as f:
        json.dump({
            "image": os.path.basename(image_path),
            "sam_latency_sec": sam_latency,
            "llm_latency_sec": llm_latency,
            "total_latency_sec": sam_latency + llm_latency,
            "detections": objects,
            "threat_assessment": llm_parsed
        }, f, indent=4)

    return sam_latency, llm_latency

# ----------------------------
# PROCESS ALL IMAGES
# ----------------------------

summary = []

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    image_path = os.path.join(IMAGE_FOLDER, filename)
    print(f"Processing {filename}...")
    
    sam_latency, llm_latency = process_image(image_path)

    # Read back the saved JSON to extract threat info for summary
    ctx_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_context.json")
    threat_level = "UNKNOWN"
    dashboard_tag = "N/A"
    attention = False
    try:
        with open(ctx_file, "r") as cf:
            ctx = json.load(cf)
            ta = ctx.get("threat_assessment", {})
            threat_level = ta.get("threat_level", "UNKNOWN")
            dashboard_tag = ta.get("dashboard_tag", "N/A")
            attention = ta.get("requires_immediate_attention", False)
    except Exception:
        pass

    summary.append({
        "image": filename,
        "sam_latency_sec": round(sam_latency, 3),
        "llm_latency_sec": round(llm_latency, 3),
        "total_latency_sec": round(sam_latency + llm_latency, 3),
        "threat_level": threat_level,
        "dashboard_tag": dashboard_tag,
        "requires_immediate_attention": attention
    })

# ----------------------------
# SAVE SUMMARY CSV
# ----------------------------

import csv

summary_csv = os.path.join(OUTPUT_FOLDER, "summary.csv")
with open(summary_csv, "w", newline="") as csvfile:
    fieldnames = ["image", "sam_latency_sec", "llm_latency_sec", "total_latency_sec",
                  "threat_level", "dashboard_tag", "requires_immediate_attention"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary:
        writer.writerow(row)

print("All images processed! Context JSON and summary CSV saved.")