"""
Synthetic Data Pipeline for Fire/Smoke Detection
- Reads images + YOLO/VOC annotations
- Extracts objects using bbox + auto-segmentation (GrabCut + color-based)
- Applies realistic enhancements (glow, reflections, depth blur, motion blur)
- Pastes objects onto new backgrounds
- Generates new YOLO/VOC annotations

Enhanced Composition Features:
✓ Fire glow with inverse-square distance falloff
✓ Floor reflections below fire sources
✓ Color temperature matching to background lighting
✓ Smoke dissipation (fades at edges)
✓ Depth-based blur (distant objects blurrier)
✓ Atmospheric perspective (haze for distant objects)
✓ Motion blur (simulates rising flames/smoke)
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import random
from typing import List, Tuple, Optional
from datetime import datetime


# ===================== CONFIGURATION =====================
CONFIG = {
    # Input paths
    "input_images_dir": "/Users/ekta/Downloads/smoke-fire-dataset/fire-images",
    "input_annotations_dir": "/Users/ekta/Downloads/smoke-fire-dataset/fire-annotations",
    "backgrounds_dir": "/Users/ekta/Downloads/smoke-fire-dataset/Backgrounds_new",
    
    # Output paths
    "output_images_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_output/images",
    "output_annotations_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_output/annotations",
    "output_crops_dir": "/Users/ekta/Downloads/smoke-fire-dataset/synthetic_output/crops",  # Optional: save extracted crops
    
    # Generation settings
    "images_per_object": 5,  # How many synthetic images to create per extracted object
    "output_size": (1280, 720),  # Output image size (width, height)
    "scale_range": (0.3, 0.8),  # Object scale relative to background
    "save_crops": True,  # Save extracted object crops for inspection
    
    # Crop settings
    "use_segmentation": True,  # True = extract natural shape, False = rectangular crop
    "feather_edges": False,  # Soft edges for natural blending
    "feather_radius": 5,  # Larger = softer edges (good for smoke)
    
    # Annotation settings
    "input_format": "voc",  # "yolo" or "voc"
    "output_format": "voc",  # "yolo" or "voc" or "both"
    "class_names": {0: "fire", 1: "smoke"},  # YOLO class ID mapping
    
    # Enhanced composition settings (realism improvements)
    "use_enhancements": False,  # Enable/disable all enhancements
    "fire_glow": True,  # Add orange glow around fire
    "fire_glow_intensity": 0.4,  # 0.0 to 1.0
    "floor_reflection": True,  # Add reflection on floor surfaces
    "floor_reflection_strength": 0.3,  # 0.0 to 1.0
    "color_matching": True,  # Match object colors to background lighting
    "smoke_dissipation": True,  # Make smoke fade at edges
    "smoke_dissipation_amount": 0.3,  # 0.0 to 1.0
    "depth_blur": True,  # Blur distant objects
    "depth_blur_max": 5,  # Maximum blur kernel size
    "atmospheric_perspective": True,  # Add haze to distant objects
    "motion_blur": True,  # Add upward motion blur to fire/smoke
    "motion_blur_strength": 3,  # 1-5
}


# ===================== ANNOTATION PARSING =====================

def parse_yolo_txt(txt_path: str, img_width: int, img_height: int, class_names: dict = None) -> List[dict]:
    """
    Parse YOLO format .txt annotation.
    Format: class_id x_center y_center width height (all normalized 0-1)
    """
    objects = []
    
    if class_names is None:
        class_names = {0: "fire", 1: "smoke"}  # Default class mapping
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            
            # Convert normalized coordinates to pixel coordinates
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height
            
            # Convert to xmin, ymin, xmax, ymax
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            
            # Clamp to image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)
            
            objects.append({
                "name": class_names.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "bbox": (xmin, ymin, xmax, ymax),
                "img_size": (img_width, img_height)
            })
    
    return objects


def parse_voc_xml(xml_path: str) -> List[dict]:
    """Parse Pascal VOC XML and return list of objects with bbox."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    objects = []
    
    # Get image filename
    filename = root.find("filename").text
    
    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    # Get all objects
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        
        objects.append({
            "name": name,
            "bbox": (xmin, ymin, xmax, ymax),
            "filename": filename,
            "img_size": (img_width, img_height)
        })
    
    return objects


def create_yolo_txt(objects: List[dict], img_width: int, img_height: int, output_path: str):
    """
    Create YOLO format annotation file.
    Format: class_id x_center y_center width height (normalized 0-1)
    """
    lines = []
    
    for obj in objects:
        class_id = obj.get("class_id", 0)
        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]
        
        # Convert to YOLO format (normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def create_voc_xml(filename: str, img_size: Tuple[int, int, int], 
                   objects: List[dict], output_path: str):
    """Create Pascal VOC XML annotation file."""
    annotation = ET.Element("annotation")
    
    ET.SubElement(annotation, "folder").text = "synthetic_output"
    ET.SubElement(annotation, "filename").text = filename
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_size[1])
    ET.SubElement(size, "height").text = str(img_size[0])
    ET.SubElement(size, "depth").text = str(img_size[2] if len(img_size) > 2 else 3)
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj["name"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        
        bbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(obj["xmin"]))
        ET.SubElement(bbox, "ymin").text = str(int(obj["ymin"]))
        ET.SubElement(bbox, "xmax").text = str(int(obj["xmax"]))
        ET.SubElement(bbox, "ymax").text = str(int(obj["ymax"]))
    
    tree = ET.ElementTree(annotation)
    tree.write(output_path)


# ===================== ENHANCED COMPOSITION =====================

def add_realistic_fire_glow(image: np.ndarray, fire_bbox: tuple, intensity: float = 0.4) -> np.ndarray:
    """
    Add realistic fire glow with distance falloff and color temperature.
    """
    x, y, w, h = fire_bbox
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Create glow mask with distance falloff
    glow = np.zeros_like(image, dtype=np.float32)
    
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Realistic falloff (inverse square law)
    radius = max(w, h) * 1.5
    glow_mask = np.clip((radius / (dist + 1)) ** 2, 0, 1)
    glow_mask = glow_mask ** 0.5  # Adjust falloff curve
    
    # Fire color gradient (orange-red)
    glow[:, :, 2] = 255 * glow_mask * intensity  # Red
    glow[:, :, 1] = 150 * glow_mask * intensity * 0.6  # Green
    glow[:, :, 0] = 50 * glow_mask * intensity * 0.2  # Blue
    
    # Additive blending
    result = image.astype(np.float32) + glow
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def add_floor_reflection(image: np.ndarray, fire_bbox: tuple, strength: float = 0.3) -> np.ndarray:
    """
    Add fire reflection on floor/surfaces below the fire.
    """
    x, y, w, h = fire_bbox
    img_h = image.shape[0]
    
    # Only apply if fire is in upper 2/3 of image
    if y + h < img_h * 0.66:
        floor_start = y + h
        floor_region = image[floor_start:, :].copy().astype(np.float32)
        
        # Create reflection gradient (stronger closer to fire)
        gradient = np.linspace(1.0, 0.0, floor_region.shape[0]).reshape(-1, 1)
        gradient = np.tile(gradient, (1, floor_region.shape[1]))
        
        # Orange tint
        floor_region[:, :, 2] += 40 * strength * gradient  # Red
        floor_region[:, :, 1] += 20 * strength * gradient  # Green
        
        image[floor_start:, :] = np.clip(floor_region, 0, 255).astype(np.uint8)
    
    return image


def color_temperature_matching(fire_crop: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Match fire color temperature to background lighting.
    """
    # Get background color temperature
    bg_mean = background.mean(axis=(0, 1))
    
    # Adjust fire colors to match scene lighting
    fire_adjusted = fire_crop.copy().astype(np.float32)
    
    # If background is cool (blue-ish), reduce fire warmth slightly
    if bg_mean[0] > bg_mean[2]:  # More blue than red
        fire_adjusted[:, :, 0] *= 1.1  # Increase blue slightly
        fire_adjusted[:, :, 2] *= 0.95  # Decrease red slightly
    
    return np.clip(fire_adjusted, 0, 255).astype(np.uint8)


def add_smoke_dissipation(smoke_crop: np.ndarray, dissipation: float = 0.3) -> np.ndarray:
    """
    Make smoke more realistic by adding dissipation/fading at edges.
    """
    # Create radial gradient from center
    h, w = smoke_crop.shape[:2]
    Y, X = np.ogrid[:h, :w]
    
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Dissipation mask (fades at edges)
    dissipation_mask = 1.0 - (dist / max_dist) * dissipation
    dissipation_mask = np.clip(dissipation_mask, 0, 1)
    
    # Apply to alpha channel
    if smoke_crop.shape[2] == 4:
        smoke_crop[:, :, 3] = (smoke_crop[:, :, 3] * dissipation_mask).astype(np.uint8)
    
    return smoke_crop


def depth_based_blur(crop: np.ndarray, depth: float = 0.5, max_blur: int = 5) -> np.ndarray:
    """
    Apply depth-based blur (objects further away are blurrier).
    depth: 0.0 (close) to 1.0 (far)
    """
    blur_amount = int(max_blur * depth)
    if blur_amount > 0:
        blur_amount = blur_amount * 2 + 1  # Must be odd
        crop = cv2.GaussianBlur(crop, (blur_amount, blur_amount), 0)
    
    return crop


def add_atmospheric_perspective(crop: np.ndarray, depth: float = 0.5) -> np.ndarray:
    """
    Reduce contrast and add slight haze for distant objects.
    """
    if depth > 0.3:
        # Reduce contrast
        gray = crop.mean()
        crop = crop.astype(np.float32)
        crop = gray + (crop - gray) * (1 - depth * 0.5)
        
        # Add slight blue/gray tint
        crop[:, :, 0] += 20 * depth  # Blue
        crop[:, :, 1] += 15 * depth  # Green
        crop[:, :, 2] += 10 * depth  # Red
        
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    
    return crop


def add_motion_blur(fire_crop: np.ndarray, direction: str = "up", strength: int = 3) -> np.ndarray:
    """
    Add motion blur to simulate rising flames/smoke.
    """
    if strength < 1:
        return fire_crop
    
    # Create motion blur kernel
    kernel_size = strength * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size))
    
    if direction == "up":
        kernel[:, kernel_size // 2] = 1
    elif direction == "diagonal":
        np.fill_diagonal(kernel, 1)
    
    kernel = kernel / kernel.sum()
    
    # Apply blur to RGB channels only
    if fire_crop.shape[2] == 4:
        fire_crop[:, :, :3] = cv2.filter2D(fire_crop[:, :, :3], -1, kernel)
    else:
        fire_crop = cv2.filter2D(fire_crop, -1, kernel)
    
    return fire_crop


# ===================== SEGMENTATION =====================

def extract_object_grabcut(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract object using GrabCut algorithm.
    Returns: (cropped_rgba, mask)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure bbox is within image bounds
    h, w = image.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    # Add padding for GrabCut
    pad = 8
    xmin_pad = max(0, xmin - pad)
    ymin_pad = max(0, ymin - pad)
    xmax_pad = min(w, xmax + pad)
    ymax_pad = min(h, ymax + pad)
    
    # Crop region with padding
    crop = image[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()
    
    # GrabCut rectangle (relative to padded crop)
    rect = (
        xmin - xmin_pad,
        ymin - ymin_pad,
        xmax - xmin,
        ymax - ymin
    )
    
    # Initialize mask
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        # Run GrabCut
        cv2.grabCut(crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask (foreground = 1 or 3)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    except:
        # Fallback to simple crop if GrabCut fails
        mask2 = np.ones(crop.shape[:2], np.uint8) * 255
    
    # Extract the original bbox region (without padding)
    rel_xmin = xmin - xmin_pad
    rel_ymin = ymin - ymin_pad
    rel_xmax = rel_xmin + (xmax - xmin)
    rel_ymax = rel_ymin + (ymax - ymin)
    
    final_crop = crop[rel_ymin:rel_ymax, rel_xmin:rel_xmax]
    final_mask = mask2[rel_ymin:rel_ymax, rel_xmin:rel_xmax]
    
    return final_crop, final_mask


def extract_object_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract object using color-based segmentation (optimized for fire/smoke).
    Returns: (cropped_rgb, mask)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure bbox is within image bounds
    h, w = image.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    
    crop = image[ymin:ymax, xmin:xmax].copy()
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Fire/flame color ranges (red, orange, yellow)
    # Red range 1 (0-10)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    # Red range 2 (170-180)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Orange range
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Yellow range
    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White/bright (fire core)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Smoke (gray tones)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine all masks
    mask = mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_white
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # If color segmentation finds very little, fall back to full bbox
    if mask.sum() < 0.1 * mask.size * 255:
        mask = np.ones(crop.shape[:2], np.uint8) * 255
    
    return crop, mask


def combine_segmentation(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine GrabCut and color segmentation for best results.
    Returns: (cropped_rgb, mask)
    """
    crop_gc, mask_gc = extract_object_grabcut(image, bbox)
    crop_color, mask_color = extract_object_color(image, bbox)
    
    # Ensure same size
    if crop_gc.shape != crop_color.shape:
        crop_color = cv2.resize(crop_color, (crop_gc.shape[1], crop_gc.shape[0]))
        mask_color = cv2.resize(mask_color, (mask_gc.shape[1], mask_gc.shape[0]))
    
    # Combine masks (intersection for cleaner result, or union for more coverage)
    combined_mask = cv2.bitwise_and(mask_gc, mask_color)
    
    # If combined is too small, use GrabCut mask
    if combined_mask.sum() < 0.1 * mask_gc.sum():
        combined_mask = mask_gc
    
    return crop_gc, combined_mask


def feather_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Apply feathering to mask edges for smoother blending."""
    # Blur the mask edges
    feathered = cv2.GaussianBlur(mask.astype(np.float32), (radius * 2 + 1, radius * 2 + 1), 0)
    return feathered.astype(np.uint8)


def create_rgba_crop(crop: np.ndarray, mask: np.ndarray, feather: bool = True) -> np.ndarray:
    """Create RGBA image from crop and mask."""
    if feather:
        mask = feather_mask(mask, CONFIG["feather_radius"])
    
    # Create RGBA
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    
    return rgba


# ===================== COMPOSITING =====================

def paste_object_on_background(background: np.ndarray, obj_rgba: np.ndarray,
                                position: Tuple[int, int], scale: float) -> Tuple[np.ndarray, dict]:
    """
    Paste object onto background at given position and scale.
    Returns: (composite_image, bbox_dict)
    """
    bg_h, bg_w = background.shape[:2]
    
    # Scale object
    new_h = int(obj_rgba.shape[0] * scale)
    new_w = int(obj_rgba.shape[1] * scale)
    
    # Ensure minimum size
    new_h = max(20, new_h)
    new_w = max(20, new_w)
    
    obj_scaled = cv2.resize(obj_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    x, y = position
    
    # Ensure object fits within background
    if x + new_w > bg_w:
        x = bg_w - new_w
    if y + new_h > bg_h:
        y = bg_h - new_h
    x = max(0, x)
    y = max(0, y)
    
    # Actual dimensions after clipping
    paste_w = min(new_w, bg_w - x)
    paste_h = min(new_h, bg_h - y)
    
    # Check if using alpha channel (segmentation enabled)
    has_alpha = obj_scaled.shape[2] == 4
    
    result = background.copy()
    
    if has_alpha:
        # Alpha blending for natural edges
        alpha = obj_scaled[:paste_h, :paste_w, 3:4].astype(np.float32) / 255.0
        obj_rgb = obj_scaled[:paste_h, :paste_w, :3].astype(np.float32)
        bg_region = result[y:y+paste_h, x:x+paste_w].astype(np.float32)
        
        blended = bg_region * (1 - alpha) + obj_rgb * alpha
        result[y:y+paste_h, x:x+paste_w] = blended.astype(np.uint8)
        
        # Calculate tight bbox from alpha mask
        alpha_2d = obj_scaled[:paste_h, :paste_w, 3]
        non_zero = np.where(alpha_2d > 25)
        
        if len(non_zero[0]) > 0:
            bbox = {
                "xmin": x + non_zero[1].min(),
                "ymin": y + non_zero[0].min(),
                "xmax": x + non_zero[1].max(),
                "ymax": y + non_zero[0].max(),
            }
        else:
            bbox = {"xmin": x, "ymin": y, "xmax": x + paste_w, "ymax": y + paste_h}
    else:
        # Direct paste (no alpha)
        result[y:y+paste_h, x:x+paste_w] = obj_scaled[:paste_h, :paste_w, :3]
        bbox = {"xmin": x, "ymin": y, "xmax": x + paste_w, "ymax": y + paste_h}
    
    return result, bbox


def generate_random_position(bg_size: Tuple[int, int], obj_size: Tuple[int, int]) -> Tuple[int, int]:
    """Generate random position for object placement."""
    bg_h, bg_w = bg_size
    obj_h, obj_w = obj_size
    
    # Prefer placing fire in lower 2/3 of image (more realistic)
    x = random.randint(0, max(0, bg_w - obj_w))
    y = random.randint(bg_h // 3, max(bg_h // 3, bg_h - obj_h))
    
    return x, y


# ===================== MAIN PIPELINE =====================

def process_single_image(image_path: str, annotation_path: str, backgrounds: List[np.ndarray]) -> int:
    """Process a single image and its annotations."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ⚠️  Could not load image: {image_path}")
        return 0
    
    img_h, img_w = image.shape[:2]
    
    # Parse annotations based on format
    if CONFIG["input_format"] == "yolo":
        objects = parse_yolo_txt(annotation_path, img_w, img_h, CONFIG["class_names"])
    else:  # voc
        objects = parse_voc_xml(annotation_path)
    
    if not objects:
        print(f"  ⚠️  No objects found in: {annotation_path}")
        return 0
    
    generated_count = 0
    base_name = Path(image_path).stem
    
    for obj_idx, obj in enumerate(objects):
        bbox = obj["bbox"]
        class_name = obj["name"]
        
        print(f"    Processing object {obj_idx + 1}: {class_name} at {bbox}")

        if class_name.lower() == "smoke":
            print(f"    Skipping smoke object (fire-only mode)")
            continue

        # Extract object with segmentation
        if CONFIG["use_segmentation"]:
            # Use segmentation to get natural shape
            crop, mask = extract_object_grabcut(image, bbox)
            
            # Apply feathering for soft edges
            if CONFIG["feather_edges"]:
                mask = feather_mask(mask, CONFIG["feather_radius"])
            
            # Create RGBA with transparency
            obj_rgba = create_rgba_crop(crop, mask, feather=False)  # Already feathered
        else:
            # Simple rectangular crop
            xmin, ymin, xmax, ymax = bbox
            h, w = image.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            
            crop = image[ymin:ymax, xmin:xmax].copy()
            
            # Convert to RGBA (fully opaque)
            obj_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            obj_rgba[:, :, 3] = 255  # Fully opaque
        
        # Save crop for inspection
        if CONFIG["save_crops"]:
            crop_path = Path(CONFIG["output_crops_dir"]) / f"{base_name}_obj{obj_idx}_{class_name}.png"
            cv2.imwrite(str(crop_path), obj_rgba)
        
        # Generate synthetic images
        for bg_idx in range(CONFIG["images_per_object"]):
            # Select random background
            bg = random.choice(backgrounds).copy()
            bg = cv2.resize(bg, CONFIG["output_size"])
            
            # Random scale and position
            scale = random.uniform(*CONFIG["scale_range"])
            scaled_h = int(obj_rgba.shape[0] * scale)
            scaled_w = int(obj_rgba.shape[1] * scale)
            position = generate_random_position(bg.shape[:2], (scaled_h, scaled_w))
            
            # ============ APPLY ENHANCEMENTS (BEFORE COMPOSITING) ============
            obj_rgba_enhanced = obj_rgba.copy()
            
            if CONFIG["use_enhancements"]:
                # Calculate depth based on vertical position (0=close, 1=far)
                depth = position[1] / bg.shape[0]
                
                # 1. Color temperature matching
                if CONFIG["color_matching"]:
                    if obj_rgba_enhanced.shape[2] == 4:
                        obj_rgba_enhanced[:, :, :3] = color_temperature_matching(obj_rgba_enhanced[:, :, :3], bg)
                    else:
                        obj_rgba_enhanced = color_temperature_matching(obj_rgba_enhanced, bg)
                
                # 2. Smoke-specific: dissipation
                if CONFIG["smoke_dissipation"] and class_name == "smoke":
                    obj_rgba_enhanced = add_smoke_dissipation(obj_rgba_enhanced, CONFIG["smoke_dissipation_amount"])
                
                # 3. Depth-based blur
                if CONFIG["depth_blur"]:
                    obj_rgba_enhanced = depth_based_blur(obj_rgba_enhanced, depth, CONFIG["depth_blur_max"])
                
                # 4. Atmospheric perspective
                if CONFIG["atmospheric_perspective"]:
                    obj_rgba_enhanced = add_atmospheric_perspective(obj_rgba_enhanced, depth)
                
                # 5. Motion blur
                if CONFIG["motion_blur"]:
                    obj_rgba_enhanced = add_motion_blur(obj_rgba_enhanced, direction="up", strength=CONFIG["motion_blur_strength"])
            
            # Composite
            composite, new_bbox = paste_object_on_background(bg, obj_rgba_enhanced, position, scale)
            new_bbox["name"] = class_name
            new_bbox["class_id"] = obj.get("class_id", 0)
            
            # ============ APPLY ENHANCEMENTS (AFTER COMPOSITING) ============
            if CONFIG["use_enhancements"]:
                # 6. Fire glow (only for fire objects)
                if CONFIG["fire_glow"] and class_name == "fire":
                    fire_bbox = (new_bbox["xmin"], new_bbox["ymin"], 
                                new_bbox["xmax"] - new_bbox["xmin"], 
                                new_bbox["ymax"] - new_bbox["ymin"])
                    composite = add_realistic_fire_glow(composite, fire_bbox, CONFIG["fire_glow_intensity"])
                
                # 7. Floor reflection (only for fire objects)
                if CONFIG["floor_reflection"] and class_name == "fire":
                    fire_bbox = (new_bbox["xmin"], new_bbox["ymin"], 
                                new_bbox["xmax"] - new_bbox["xmin"], 
                                new_bbox["ymax"] - new_bbox["ymin"])
                    composite = add_floor_reflection(composite, fire_bbox, CONFIG["floor_reflection_strength"])
            
            # Generate output filename
            timestamp = datetime.now().strftime("%H%M%S%f")[:10]
            output_name = f"{base_name}_obj{obj_idx}_bg{bg_idx}_{timestamp}"
            
            # Save image
            img_path = Path(CONFIG["output_images_dir"]) / f"{output_name}.jpg"
            cv2.imwrite(str(img_path), composite)
            
            # Save annotations in specified format(s)
            if CONFIG["output_format"] in ["yolo", "both"]:
                txt_path_out = Path(CONFIG["output_annotations_dir"]) / f"{output_name}.txt"
                create_yolo_txt([new_bbox], CONFIG["output_size"][0], CONFIG["output_size"][1], str(txt_path_out))
            
            if CONFIG["output_format"] in ["voc", "both"]:
                xml_path_out = Path(CONFIG["output_annotations_dir"]) / f"{output_name}.xml"
                create_voc_xml(
                    f"{output_name}.jpg",
                    composite.shape,
                    [new_bbox],
                    str(xml_path_out)
                )
            
            generated_count += 1
    
    return generated_count


def load_backgrounds() -> List[np.ndarray]:
    """Load all background images."""
    bg_dir = Path(CONFIG["backgrounds_dir"])
    
    if not bg_dir.exists():
        print(f"⚠️  Backgrounds directory not found: {bg_dir}")
        print("   Creating random backgrounds instead...")
        
        # Generate some random backgrounds
        backgrounds = []
        for _ in range(10):
            bg = generate_random_background(*CONFIG["output_size"])
            backgrounds.append(bg)
        return backgrounds
    
    backgrounds = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for bg_path in bg_dir.glob(ext):
            bg = cv2.imread(str(bg_path))
            if bg is not None:
                backgrounds.append(bg)
    
    if not backgrounds:
        print("⚠️  No background images found. Generating random backgrounds...")
        for _ in range(10):
            bg = generate_random_background(*CONFIG["output_size"])
            backgrounds.append(bg)
    
    return backgrounds


def generate_random_background(width: int, height: int) -> np.ndarray:
    """Generate a random background image."""
    bg_type = random.choice(["gradient", "solid", "indoor"])
    
    if bg_type == "gradient":
        top_color = np.array([random.randint(100, 200) for _ in range(3)])
        bottom_color = np.array([random.randint(50, 150) for _ in range(3)])
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            ratio = y / height
            bg[y, :] = (top_color * (1 - ratio) + bottom_color * ratio).astype(np.uint8)
    
    elif bg_type == "indoor":
        # Simple indoor scene
        wall_color = [random.randint(150, 220) for _ in range(3)]
        floor_color = [max(0, c - 50) for c in wall_color]
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        split = int(height * 0.5)
        bg[:split, :] = wall_color
        bg[split:, :] = floor_color
    
    else:
        color = [random.randint(50, 200) for _ in range(3)]
        bg = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Add noise
    noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return bg


def run_pipeline():
    """Main pipeline execution."""
    
    # Create output directories
    Path(CONFIG["output_images_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["output_annotations_dir"]).mkdir(parents=True, exist_ok=True)
    if CONFIG["save_crops"]:
        Path(CONFIG["output_crops_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Load backgrounds
    print("📂 Loading backgrounds...")
    backgrounds = load_backgrounds()
    print(f"   Loaded {len(backgrounds)} backgrounds")
    
    # Get input images and annotations
    images_dir = Path(CONFIG["input_images_dir"])
    annotations_dir = Path(CONFIG["input_annotations_dir"])
    
    if not images_dir.exists():
        print(f"❌ Input images directory not found: {images_dir}")
        print("   Please create the directory and add your images.")
        return
    
    if not annotations_dir.exists():
        print(f"❌ Input annotations directory not found: {annotations_dir}")
        print("   Please create the directory and add your XML files.")
        return
    
    # Find all image-annotation pairs
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    # Determine annotation extension based on input format
    ann_ext = ".txt" if CONFIG["input_format"] == "yolo" else ".xml"
    
    print(f"\n🔥 Starting synthetic data generation...")
    print(f"   Input format: {CONFIG['input_format'].upper()}")
    print(f"   Output format: {CONFIG['output_format'].upper()}")
    print(f"   Input images: {len(image_files)}")
    print(f"   Images per object: {CONFIG['images_per_object']}")
    print(f"   Output size: {CONFIG['output_size']}")
    
    total_generated = 0
    
    for img_path in image_files:
        # Find corresponding annotation
        ann_path = annotations_dir / f"{img_path.stem}{ann_ext}"
        
        if not ann_path.exists():
            print(f"\n⚠️  No annotation found for: {img_path.name} (expected: {ann_path.name})")
            continue
        
        print(f"\n📄 Processing: {img_path.name}")
        
        count = process_single_image(str(img_path), str(ann_path), backgrounds)
        total_generated += count
        print(f"   Generated {count} synthetic images")
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Total images generated: {total_generated}")
    print(f"   Output images: {CONFIG['output_images_dir']}")
    print(f"   Output annotations: {CONFIG['output_annotations_dir']}")
    if CONFIG["save_crops"]:
        print(f"   Extracted crops: {CONFIG['output_crops_dir']}")


if __name__ == "__main__":
    """
    ============================================================================
    USAGE GUIDE - Enhanced Synthetic Data Pipeline
    ============================================================================
    
    QUICK START:
    1. Set your paths in CONFIG above (lines 20-46)
    2. Run: python synthetic_pipeline.py
    
    ENHANCEMENT SETTINGS (Toggle On/Off):
    
    Basic Enhancements:
    - "use_enhancements": True/False     → Master switch for all enhancements
    - "fire_glow": True/False            → Adds orange glow around fire
    - "floor_reflection": True/False     → Adds reflection on floor surfaces
    - "color_matching": True/False       → Matches object colors to background
    
    Smoke Improvements:
    - "smoke_dissipation": True/False    → Makes smoke fade naturally at edges
    - "smoke_dissipation_amount": 0.3    → 0.0 (none) to 1.0 (heavy fade)
    
    Depth Realism:
    - "depth_blur": True/False           → Blurs distant objects
    - "depth_blur_max": 5                → Maximum blur amount
    - "atmospheric_perspective": True    → Adds haze to distant objects
    
    Motion Effects:
    - "motion_blur": True/False          → Adds upward blur for rising flames
    - "motion_blur_strength": 3          → 1 (subtle) to 5 (strong)
    
    RECOMMENDED PRESETS:
    
    Maximum Realism (slow, best quality):
        "use_enhancements": True
        All enhancement options: True
        
    Fast Generation (no enhancements):
        "use_enhancements": False
        
    Balanced (recommended):
        "use_enhancements": True
        "fire_glow": True
        "color_matching": True
        "smoke_dissipation": True
        "depth_blur": False           ← Disable for speed
        "motion_blur": False          ← Disable for speed
    
    BEFORE vs AFTER:
    
    Without Enhancements:                With Enhancements:
    ┌─────────────────────┐             ┌─────────────────────┐
    │                     │             │   ░░░               │
    │       🔥            │      →      │  ░🔥░  ← Glow       │
    │    (hard edges)     │             │   ░░░               │
    ├─────────────────────┤             ├─────────────────────┤
    │   Gray floor        │             │  🟠 Orange tint     │
    │                     │             │   (reflection)      │
    └─────────────────────┘             └─────────────────────┘
    Looks pasted                        Looks realistic
    
    ============================================================================
    """
    run_pipeline()

