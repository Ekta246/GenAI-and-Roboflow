from roboflow import Roboflow
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Prefer dataset-specific key, fallback to general Roboflow key
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_DATASET_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "Missing Roboflow key. Set ROBOFLOW_DATASET_API_KEY "
        "(or ROBOFLOW_API_KEY) in your environment or .env file."
    )

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")
dataset = project.version(2).download("yolov11")

