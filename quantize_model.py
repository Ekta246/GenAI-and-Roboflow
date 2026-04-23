"""
Model Quantization Script
Supports multiple quantization methods for object detection models:

  1. YOLO → ONNX export with INT8/FP16 quantization
  2. ONNX Runtime static quantization (any ONNX model)
  3. PyTorch dynamic quantization (any .pt/.pth model)
  4. TensorRT FP16/INT8 (NVIDIA GPU required)

Usage:
  python quantize_model.py
  Adjust CONFIG below to choose method and model path.
"""

import os
import time
import shutil
from pathlib import Path


# ===================== CONFIGURATION =====================
CONFIG = {
    # Input model path (.pt for YOLO/PyTorch, .onnx for ONNX models)
    "model_path": "/Users/ekta/Downloads/yolo11-d-fire-dataset.pt",

    # Output directory for quantized models
    "output_dir": "/Users/ekta/Downloads/smoke-fire-dataset/quantized_models",

    # Quantization method: "yolo_export", "onnx_static", "pytorch_dynamic", "tensorrt"
    "method": "yolo_export",

    # --- YOLO export settings ---
    "yolo_format": "onnx",        # "onnx", "torchscript", "openvino", "tflite", "engine" (TensorRT)
    "yolo_half": True,             # FP16 quantization (GPU export)
    "yolo_int8": False,            # INT8 quantization (needs calibration data)
    "yolo_imgsz": 640,             # Input image size
    "yolo_dynamic": False,         # Dynamic input shapes
    "yolo_simplify": True,         # Simplify ONNX graph
    "yolo_opset": 17,              # ONNX opset version

    # --- ONNX static quantization settings ---
    "onnx_quant_type": "int8",     # "int8" or "uint8"
    "calibration_images_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_data/sf-images/images",
    "calibration_samples": 50,     # Number of images for calibration

    # --- Benchmark settings ---
    "run_benchmark": True,         # Compare original vs quantized speed
    "benchmark_iterations": 100,
    "benchmark_image": None,       # None = use first image from calibration dir
}


def quantize_yolo_export():
    """
    Export YOLO model to optimized format with quantization.
    Supports: ONNX (FP16/INT8), TorchScript, OpenVINO, TFLite, TensorRT.
    """
    from ultralytics import YOLO

    model = YOLO(CONFIG["model_path"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = CONFIG["yolo_format"]
    half = CONFIG["yolo_half"]
    int8 = CONFIG["yolo_int8"]

    print(f"\n  YOLO Export Quantization")
    print(f"  Model:  {CONFIG['model_path']}")
    print(f"  Format: {fmt}")
    print(f"  FP16:   {half}")
    print(f"  INT8:   {int8}")
    print(f"  ImgSz:  {CONFIG['yolo_imgsz']}")
    print(f"{'=' * 50}")

    export_args = {
        "format": fmt,
        "imgsz": CONFIG["yolo_imgsz"],
        "half": half,
        "int8": int8,
        "simplify": CONFIG["yolo_simplify"],
        "dynamic": CONFIG["yolo_dynamic"],
    }

    if fmt == "onnx":
        export_args["opset"] = CONFIG["yolo_opset"]

    if int8 and CONFIG["calibration_images_dir"]:
        export_args["data"] = CONFIG["calibration_images_dir"]

    t0 = time.time()
    exported_path = model.export(**export_args)
    elapsed = time.time() - t0

    print(f"\n  Export completed in {elapsed:.1f}s")
    print(f"  Exported to: {exported_path}")

    if exported_path and os.path.exists(exported_path):
        src = Path(exported_path)
        dst = output_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

        orig_size = os.path.getsize(CONFIG["model_path"]) / (1024 * 1024)
        if src.is_file():
            new_size = os.path.getsize(str(dst)) / (1024 * 1024)
        else:
            new_size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / (1024 * 1024)

        print(f"\n  Size comparison:")
        print(f"    Original (.pt):   {orig_size:.1f} MB")
        print(f"    Quantized ({fmt}): {new_size:.1f} MB")
        print(f"    Reduction:        {(1 - new_size/orig_size)*100:.1f}%")
        print(f"  Saved to: {dst}")

    return exported_path


def quantize_onnx_static():
    """
    Static INT8 quantization of an ONNX model using ONNX Runtime.
    Requires calibration data for best accuracy.
    """
    import numpy as np
    import cv2
    import onnx
    from onnxruntime.quantization import (
        quantize_static,
        quantize_dynamic,
        CalibrationDataReader,
        QuantType,
        QuantFormat,
    )

    model_path = CONFIG["model_path"]
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.endswith(".onnx"):
        print("  ONNX static quantization requires .onnx input.")
        print("  Run with method='yolo_export' first to get an ONNX model,")
        print("  then point model_path to the .onnx file.")
        return None

    print(f"\n  ONNX Static INT8 Quantization")
    print(f"  Model: {model_path}")
    print(f"{'=' * 50}")

    class FireSmokeCalibrationReader(CalibrationDataReader):
        def __init__(self, calib_dir, input_name, input_shape, num_samples):
            self.data = []
            img_dir = Path(calib_dir)
            images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))[:num_samples]

            _, c, h, w = input_shape
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (w, h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                self.data.append({input_name: img})

            self.idx = 0
            print(f"  Loaded {len(self.data)} calibration images")

        def get_next(self):
            if self.idx >= len(self.data):
                return None
            result = self.data[self.idx]
            self.idx += 1
            return result

    onnx_model = onnx.load(model_path)
    input_info = onnx_model.graph.input[0]
    input_name = input_info.name
    input_shape = [d.dim_value for d in input_info.type.tensor_type.shape.dim]
    if input_shape[0] == 0:
        input_shape[0] = 1
    print(f"  Input: {input_name} shape={input_shape}")

    calibrator = FireSmokeCalibrationReader(
        CONFIG["calibration_images_dir"],
        input_name,
        input_shape,
        CONFIG["calibration_samples"],
    )

    stem = Path(model_path).stem
    output_path = str(output_dir / f"{stem}_int8.onnx")

    t0 = time.time()
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibrator,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    elapsed = time.time() - t0

    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n  Quantization completed in {elapsed:.1f}s")
    print(f"  Original:  {orig_size:.1f} MB")
    print(f"  Quantized: {new_size:.1f} MB")
    print(f"  Reduction: {(1 - new_size/orig_size)*100:.1f}%")
    print(f"  Saved to:  {output_path}")

    return output_path


def quantize_pytorch_dynamic():
    """
    PyTorch dynamic quantization.
    Quantizes weights to INT8, activations remain FP32.
    Good for CPU inference with minimal accuracy loss.
    """
    import torch

    model_path = CONFIG["model_path"]
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  PyTorch Dynamic Quantization")
    print(f"  Model: {model_path}")
    print(f"{'=' * 50}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
    elif hasattr(checkpoint, "model"):
        model = checkpoint.model
    else:
        model = checkpoint

    if hasattr(model, "float"):
        model = model.float()
    model.eval()

    t0 = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
    )
    elapsed = time.time() - t0

    stem = Path(model_path).stem
    output_path = str(output_dir / f"{stem}_dynamic_int8.pt")

    if isinstance(checkpoint, dict):
        checkpoint["model"] = quantized_model
        torch.save(checkpoint, output_path)
    else:
        torch.save(quantized_model, output_path)

    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n  Quantization completed in {elapsed:.1f}s")
    print(f"  Original:  {orig_size:.1f} MB")
    print(f"  Quantized: {new_size:.1f} MB")
    print(f"  Reduction: {(1 - new_size/orig_size)*100:.1f}%")
    print(f"  Saved to:  {output_path}")

    return output_path


def quantize_tensorrt():
    """
    Export to TensorRT engine with FP16/INT8.
    Requires NVIDIA GPU + TensorRT installed.
    Uses YOLO's built-in TensorRT export.
    """
    from ultralytics import YOLO

    model = YOLO(CONFIG["model_path"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  TensorRT Export")
    print(f"  Model: {CONFIG['model_path']}")
    print(f"  FP16:  {CONFIG['yolo_half']}")
    print(f"  INT8:  {CONFIG['yolo_int8']}")
    print(f"{'=' * 50}")

    export_args = {
        "format": "engine",
        "imgsz": CONFIG["yolo_imgsz"],
        "half": CONFIG["yolo_half"],
        "int8": CONFIG["yolo_int8"],
        "dynamic": CONFIG["yolo_dynamic"],
    }

    if CONFIG["yolo_int8"] and CONFIG["calibration_images_dir"]:
        export_args["data"] = CONFIG["calibration_images_dir"]

    t0 = time.time()
    exported_path = model.export(**export_args)
    elapsed = time.time() - t0

    print(f"\n  Export completed in {elapsed:.1f}s")
    print(f"  Exported to: {exported_path}")

    if exported_path and os.path.exists(exported_path):
        src = Path(exported_path)
        dst = output_dir / src.name
        shutil.copy2(str(src), str(dst))

        orig_size = os.path.getsize(CONFIG["model_path"]) / (1024 * 1024)
        new_size = os.path.getsize(str(dst)) / (1024 * 1024)
        print(f"  Original: {orig_size:.1f} MB")
        print(f"  Engine:   {new_size:.1f} MB")
        print(f"  Saved to: {dst}")

    return exported_path


def run_benchmark(original_path, quantized_path):
    """Benchmark original vs quantized model inference speed."""
    import cv2
    import numpy as np

    calib_dir = Path(CONFIG["calibration_images_dir"])
    images = sorted(list(calib_dir.glob("*.png")) + list(calib_dir.glob("*.jpg")))
    if CONFIG["benchmark_image"]:
        test_img_path = CONFIG["benchmark_image"]
    elif images:
        test_img_path = str(images[0])
    else:
        print("  No images found for benchmark")
        return

    iterations = CONFIG["benchmark_iterations"]
    print(f"\n{'=' * 50}")
    print(f"  Benchmark: {iterations} iterations")
    print(f"  Image: {os.path.basename(test_img_path)}")
    print(f"{'=' * 50}")

    if original_path.endswith(".pt"):
        from ultralytics import YOLO

        # Original
        model_orig = YOLO(original_path)
        model_orig.predict(test_img_path, verbose=False)  # warmup
        t0 = time.time()
        for _ in range(iterations):
            model_orig.predict(test_img_path, verbose=False)
        orig_time = (time.time() - t0) / iterations * 1000

        # Quantized
        if quantized_path and os.path.exists(quantized_path):
            model_quant = YOLO(quantized_path)
            model_quant.predict(test_img_path, verbose=False)  # warmup
            t0 = time.time()
            for _ in range(iterations):
                model_quant.predict(test_img_path, verbose=False)
            quant_time = (time.time() - t0) / iterations * 1000

            print(f"\n  Original:  {orig_time:.1f} ms/image")
            print(f"  Quantized: {quant_time:.1f} ms/image")
            print(f"  Speedup:   {orig_time/quant_time:.2f}x")
        else:
            print(f"  Original: {orig_time:.1f} ms/image")
            print(f"  (Quantized model not found for comparison)")


# ===================== MAIN =====================

METHODS = {
    "yolo_export": quantize_yolo_export,
    "onnx_static": quantize_onnx_static,
    "pytorch_dynamic": quantize_pytorch_dynamic,
    "tensorrt": quantize_tensorrt,
}

if __name__ == "__main__":
    method = CONFIG["method"]
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║   Model Quantization                                       ║
    ║   Method: {method:<48s}║
    ╚════════════════════════════════════════════════════════════╝
    """)

    if method not in METHODS:
        print(f"  Unknown method '{method}'. Choose from: {list(METHODS.keys())}")
    else:
        quantized_path = METHODS[method]()

        if CONFIG["run_benchmark"] and quantized_path:
            run_benchmark(CONFIG["model_path"], quantized_path)

    print("\nDone.")
