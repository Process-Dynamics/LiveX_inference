#!/usr/bin/env python3
"""
Batch Image Processing Script
Process a folder of frame images with flat field correction, 
linear stretch, and Gaussian filtering.
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter
import re

# ============ CONFIGURATION ============
INPUT_DIR = "/path/to/input/folder"
OUTPUT_DIR = "/path/to/output/folder"
START_FRAME = 100  # Can be frame number (100) or filename ("frame_0100.png")
END_FRAME = None  # None = process to the last frame, or frame number/filename
SIGMA = 1.0
# =======================================


def extract_frame_number(filename):
    """Extract frame number from filename (last 4 digits)"""
    digits = re.findall(r'\d+', filename)
    if digits:
        return int(digits[-1][-4:])
    return 0


def parse_frame_param(param):
    """Parse frame parameter - can be int or filename string"""
    if isinstance(param, int):
        return param
    elif isinstance(param, str):
        return extract_frame_number(param)
    elif param is None:
        return None
    else:
        raise ValueError(f"Invalid frame parameter: {param}")


def load_image(filepath):
    """Load image and convert to float32 [0, 1]"""
    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {filepath}")
    
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        return img.astype(np.float32)


def save_image(filepath, img, original_format):
    """Save image in the same format as input"""
    ext = original_format.lower()
    
    if ext in ['.png', '.jpg', '.jpeg']:
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(filepath), img_8bit)
    elif ext in ['.tif', '.tiff']:
        img_16bit = (np.clip(img, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(filepath), img_16bit)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def compute_flat_field(frames):
    """Compute flat field from a list of frames (mean)"""
    stacked = np.stack(frames, axis=0)
    flat = np.mean(stacked, axis=0)
    flat[flat == 0] = 1.0
    return flat


def process_single_frame(frame, flat_field, sigma):
    """
    Process a single frame:
    1. Flat field correction
    2. Linear stretch (1%/99% percentiles)
    3. Gaussian filtering
    """
    corrected = frame / flat_field
    corrected = np.clip(corrected, 0, None)
    
    p1 = np.percentile(corrected, 1)
    p99 = np.percentile(corrected, 99)
    
    if p99 > p1:
        stretched = (corrected - p1) / (p99 - p1)
    else:
        stretched = corrected
    
    stretched = np.clip(stretched, 0, 1)
    
    if sigma > 0:
        filtered = gaussian_filter(stretched, sigma=sigma)
    else:
        filtered = stretched
    
    return filtered


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    all_files = []
    for ext in extensions:
        all_files.extend(input_path.glob(f'*{ext}'))
        all_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    all_files = sorted(all_files, key=lambda x: extract_frame_number(x.name))
    
    if len(all_files) == 0:
        raise ValueError(f"No image files found in {INPUT_DIR}")
    
    print(f"Found {len(all_files)} images")
    print(f"Frame range: {extract_frame_number(all_files[0].name)} - {extract_frame_number(all_files[-1].name)}")
    
    frame_numbers = [extract_frame_number(f.name) for f in all_files]
    file_dict = {num: f for num, f in zip(frame_numbers, all_files)}
    
    start_frame = parse_frame_param(START_FRAME)
    end_frame = parse_frame_param(END_FRAME) if END_FRAME is not None else max(frame_numbers)
    
    flat_start = start_frame - 20
    flat_end = start_frame - 1
    
    print(f"\nComputing flat field from frames {flat_start} - {flat_end}...")
    flat_frames = []
    for i in range(flat_start, flat_end + 1):
        if i not in file_dict:
            raise ValueError(f"Frame {i} not found for flat field calculation")
        flat_frames.append(load_image(file_dict[i]))
    
    flat_field = compute_flat_field(flat_frames)
    print(f"Flat field computed with shape {flat_field.shape}")
    
    process_frames = [i for i in frame_numbers if start_frame <= i <= end_frame]
    
    print(f"\nProcessing frames {start_frame} - {end_frame} ({len(process_frames)} frames)...")
    print(f"Sigma: {SIGMA}")
    
    for idx, frame_num in enumerate(process_frames):
        filepath = file_dict[frame_num]
        
        frame = load_image(filepath)
        processed = process_single_frame(frame, flat_field, SIGMA)
        
        output_file = output_path / filepath.name
        save_image(output_file, processed, filepath.suffix)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(process_frames):
            print(f"  Processed {idx + 1}/{len(process_frames)} frames")
    
    print(f"\n✅ Complete! Processed {len(process_frames)} frames saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
