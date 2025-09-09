#!/usr/bin/env python3
"""
UNet++ + EfficientNet-B6 X-ray Segmentation Inference Script

Professional inference pipeline for medical image segmentation with support for:
- Nested directory structures
- Multiple output formats (probability maps, binary masks, overlay images)
- Per-experiment microstructure ratio analysis (CSV data + trend curves)
- Batch processing with AMP support
- Original image size restoration
- Comprehensive statistics and logging

Output Structure:
- experiment_A/probabilities/ : Heatmap visualizations (0.8-1.0 probability range)
- experiment_A/masks/ : Binary segmentation masks (black/white)
- experiment_A/overlay/ : Original images with red foreground overlay
- experiment_A/microstructure_analysis.csv : Per-frame microstructure ratios (frame, microstructure_ratio)
- experiment_A/microstructure_curves.png : Trend curves for this experiment only

Note: mask_ratio = foreground_pixels / total_pixels
      microstructure_ratio = 1 - mask_ratio

Author: Medical AI Team
Version: 2.0 Production - Single File Edition - Per Experiment Analysis
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import segmentation_models_pytorch as smp

# ================================================================
# CONFIGURATION PARAMETERS - Modify these for your setup
# ================================================================

# Required paths
MODEL_PATH = r"/home/shun/Project/Segmentation/UNet3Plus_test/best_model.pth"
INPUT_PATH = r"/home/shun/Project/Segmentation/UNet3Plus_test/Test_images"
OUTPUT_DIR = r"/home/shun/Project/Segmentation/UNet3Plus_test/Result"

# Model configuration
MODEL_ARCHITECTURE = "UNetPlusPlus"          # Model architecture ("UNet" or "UNetPlusPlus")
MODEL_BACKBONE = "efficientnet-b6"           # Encoder backbone
INPUT_CHANNELS = 3                           # Input channels (3 for RGB)
OUTPUT_CLASSES = 2                           # Output classes (2 for binary segmentation)

# Inference parameters
TARGET_SIZE = 1536                           # Target image size for processing
CROP_MARGIN = 0                              # Edge cropping pixels (0 = no cropping)
BATCH_SIZE = 1                               # Batch processing size
DEVICE = "cuda"                               # Processing device ("cuda" or "cpu")
RESIZE_TO_ORIGINAL = True                    # Whether to resize results back to original dimensions

# Feature toggles
USE_AMP = False                              # Use Automatic Mixed Precision (GPU only) - disabled for CPU
OUTPUT_PROBABILITIES = True                  # Generate probability heatmaps
OUTPUT_MASKS = True                          # Generate binary segmentation masks
OUTPUT_OVERLAY = True                        # Generate overlay visualizations
OUTPUT_MICROSTRUCTURE_ANALYSIS = True       # Generate microstructure ratio analysis per experiment (CSV + curves)

# Note: AMP is automatically disabled when using CPU device

# File filtering
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF']

# ================================================================
# MODEL CREATION
# ================================================================

def create_model(in_channels=3, num_classes=2, encoder_name=None, model_name=None, architecture="UNetPlusPlus"):
    """
    Create segmentation model with support for UNet and UNet++ architectures
    
    Args:
        in_channels: Number of input channels (3 for RGB images)
        num_classes: Number of output classes (2 for binary segmentation)
        encoder_name: Encoder backbone architecture name (e.g., 'efficientnet-b6')
        model_name: Legacy parameter for backward compatibility
        architecture: Model architecture ("UNet" or "UNetPlusPlus")
    
    Returns:
        model: Configured PyTorch segmentation model
    """
    
    # Handle legacy parameter naming for backward compatibility
    if model_name is not None and encoder_name is None:
        encoder_name = model_name
        print(f"Using legacy parameter 'model_name' -> 'encoder_name': {encoder_name}")
    
    # Set default encoder if none specified
    if encoder_name is None:
        encoder_name = "efficientnet-b6"
    
    # Common parameters for all architectures
    common_params = {
        'encoder_name': encoder_name,
        'encoder_weights': "imagenet",  # Use ImageNet pretrained weights
        'in_channels': in_channels,
        'classes': num_classes,
        'encoder_depth': 5,  # Standard depth for EfficientNet encoders
    }
    
    # Create model based on specified architecture
    if architecture.lower() == "unet":
        model = smp.Unet(
            **common_params,
            decoder_channels=(512, 256, 128, 64, 32),
        )
        print(f"Created UNet model with {encoder_name} backbone")
    elif architecture.lower() == "unetplusplus":
        model = smp.UnetPlusPlus(
            **common_params,
            decoder_channels=(512, 256, 128, 64, 32),
        )
        print(f"Created UNet++ model with {encoder_name} backbone")
    else:
        # Default to UNet++ for unsupported architectures
        print(f"Warning: Unsupported architecture '{architecture}'. Using UNetPlusPlus instead.")
        print("Supported architectures: UNet, UNetPlusPlus")
        
        model = smp.UnetPlusPlus(
            **common_params,
            decoder_channels=(512, 256, 128, 64, 32),
        )
        print(f"Created UNet++ model (default) with {encoder_name} backbone")
    
    print(f"Input channels: {in_channels}, Output classes: {num_classes}")
    
    return model

# ================================================================
# IMAGE PROCESSING UTILITIES
# ================================================================

def inverse_transform(result, transform_params):
    """
    Transform inference results from target_size back to original image dimensions
    
    This function reverses the preprocessing transformations to restore the
    segmentation results to the original image size and aspect ratio.
    
    Args:
        result: numpy array [target_size, target_size] - processed result
        transform_params: dict - transformation parameters from preprocessing
    
    Returns:
        restored_result: numpy array [original_height, original_width] - restored result
    """
    # Extract transformation parameters
    original_height = transform_params['original_height']
    original_width = transform_params['original_width']
    scale = transform_params['scale']
    new_height = transform_params['new_height']
    new_width = transform_params['new_width']
    pad_top = transform_params['pad_top']
    pad_left = transform_params['pad_left']
    
    # Step 1: Remove padding applied during preprocessing
    unpadded = result[pad_top:pad_top+new_height, pad_left:pad_left+new_width]
    
    # Step 2: Resize back to original dimensions
    if result.dtype == np.float32 or result.dtype == np.float64:
        # For probability maps - use bilinear interpolation for smooth gradients
        restored = cv2.resize(unpadded, (original_width, original_height), 
                            interpolation=cv2.INTER_LINEAR)
    else:
        # For masks - use bilinear then threshold for smoother edges than nearest neighbor
        restored = cv2.resize(unpadded.astype(np.float32), (original_width, original_height), 
                            interpolation=cv2.INTER_LINEAR)
        # Threshold back to binary values
        restored = (restored > 0.5).astype(np.int64)
    
    return restored

# ================================================================
# DATASET AND DATA LOADING
# ================================================================

class XRayInferenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for X-ray image inference with proper preprocessing pipeline
    
    Supports nested directory structures and applies consistent preprocessing:
    1. Optional edge cropping
    2. Aspect-ratio preserving resize
    3. Square padding with zero values
    4. Grayscale to RGB conversion
    5. ImageNet normalization
    """
    
    def __init__(self, image_info_list, target_size=1536, crop_margin=0):
        """
        Initialize inference dataset
        
        Args:
            image_info_list: List of dicts with 'path' and 'relative_path' keys
            target_size: Target square size for processing
            crop_margin: Pixels to crop from each edge (0 = no cropping)
        """
        self.image_info_list = image_info_list
        self.target_size = target_size
        self.crop_margin = crop_margin
        
        # Build preprocessing pipeline
        transforms = []
        
        # Optional edge cropping
        if self.crop_margin > 0:
            transforms.append(
                A.Crop(self.crop_margin, self.crop_margin, 
                      -self.crop_margin, -self.crop_margin, always_apply=True)
            )
        
        # Core preprocessing pipeline
        transforms.extend([
            # Resize maintaining aspect ratio
            A.LongestMaxSize(max_size=self.target_size, interpolation=cv2.INTER_LINEAR),
            
            # Pad to square with zero padding
            A.PadIfNeeded(self.target_size, self.target_size,
                        border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Convert grayscale to 3-channel RGB
            A.Lambda(image=lambda x, **k: np.repeat(x[..., None], 3, axis=2)),
            
            # ImageNet normalization
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       max_pixel_value=255.0),
            
            # Convert to tensor
            ToTensorV2()
        ])
        
        self.transform = A.Compose(transforms)
    
    def __len__(self):
        return len(self.image_info_list)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single image for inference
        
        Returns:
            dict: Contains processed image tensor and transformation parameters
        """
        image_info = self.image_info_list[idx]
        img_path = image_info['path']
        relative_path = image_info['relative_path']
        
        # Load image as grayscale
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        # Apply optional edge cropping
        if self.crop_margin > 0:
            image = image[self.crop_margin:-self.crop_margin, self.crop_margin:-self.crop_margin]
        
        # Store original dimensions (after potential cropping)
        original_height, original_width = image.shape
        
        # Calculate resize parameters for inverse transformation
        scale = self.target_size / max(original_height, original_width)
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        
        # Calculate padding parameters
        pad_top = (self.target_size - new_height) // 2
        pad_bottom = self.target_size - new_height - pad_top
        pad_left = (self.target_size - new_width) // 2
        pad_right = self.target_size - new_width - pad_left
        
        # Apply preprocessing transformations
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]
        
        # Return data with individual transform parameters (avoids DataLoader serialization issues)
        return {
            "image": image_tensor,
            "img_path": str(img_path),
            "relative_path": relative_path,
            "filename": os.path.basename(img_path),
            # Transform parameters for inverse transformation
            "original_height": original_height,
            "original_width": original_width,
            "scale": scale,
            "new_height": new_height,
            "new_width": new_width,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right
        }

# ================================================================
# MAIN INFERENCE CLASS
# ================================================================

class XRayInference:
    """
    Professional X-ray segmentation inference pipeline
    
    Features:
    - Model loading with proper state dict handling
    - Batch processing with AMP support
    - Multiple output format generation
    - Per-experiment microstructure ratio analysis
    - Comprehensive statistics and logging
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Processing device (auto-detected if None)
        """
        self.model_path = model_path
        
        # Setup processing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load and setup model
        self.model = self._load_model()
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _load_model(self):
        """
        Load trained model with proper checkpoint handling
        
        Returns:
            model: Loaded PyTorch model ready for inference
        """
        print(f"Loading model from: {self.model_path}")
        
        # Create model architecture
        model = create_model(
            in_channels=INPUT_CHANNELS,
            num_classes=OUTPUT_CLASSES,
            encoder_name=MODEL_BACKBONE,
            architecture=MODEL_ARCHITECTURE
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove potential 'module.' prefix from DataParallel training
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value 
                         for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
    
    def predict_single(self, image_path, use_amp=USE_AMP):
        """
        Perform inference on a single image
        
        Args:
            image_path: Path to input image
            use_amp: Use Automatic Mixed Precision
        
        Returns:
            dict: Inference results with prediction, probabilities, and metadata
        """
        # Create single image dataset
        image_info = {'path': image_path, 'relative_path': os.path.basename(image_path)}
        dataset = XRayInferenceDataset([image_info], 
                                     target_size=TARGET_SIZE,
                                     crop_margin=CROP_MARGIN)
        
        sample = dataset[0]
        image_tensor = sample['image'].unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Build transform parameters dictionary
        transform_params = {
            'original_height': sample['original_height'],
            'original_width': sample['original_width'],
            'scale': sample['scale'],
            'new_height': sample['new_height'],
            'new_width': sample['new_width'],
            'pad_top': sample['pad_top'],
            'pad_bottom': sample['pad_bottom'],
            'pad_left': sample['pad_left'],
            'pad_right': sample['pad_right']
        }
        
        # Perform inference
        with torch.no_grad():
            if use_amp and self.device.type == 'cuda':
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(image_tensor)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(image_tensor)
            else:
                # CPU inference or AMP disabled
                outputs = self.model(image_tensor)
        
        # Process outputs
        probabilities = torch.softmax(outputs, dim=1)  # [1, 2, H, W]
        predictions = torch.argmax(outputs, dim=1)     # [1, H, W]
        
        # Convert to numpy
        pred_mask = predictions[0].cpu().numpy()       # [H, W]
        prob_maps = probabilities[0].cpu().numpy()     # [2, H, W]
        confidence = prob_maps[1]  # Foreground confidence
        
        print(f"Single image inference: original_shape=({transform_params['original_height']}, {transform_params['original_width']}), pred_shape={pred_mask.shape}")
        
        # Restore to original dimensions if requested
        if RESIZE_TO_ORIGINAL:
            pred_mask = inverse_transform(pred_mask, transform_params)
            confidence = inverse_transform(confidence, transform_params)
            # Process full probability maps
            prob_maps_resized = np.zeros((2, transform_params['original_height'], transform_params['original_width']))
            for i in range(2):
                prob_maps_resized[i] = inverse_transform(prob_maps[i], transform_params)
            prob_maps = prob_maps_resized
        
        return {
            'prediction': pred_mask,
            'probabilities': prob_maps,
            'confidence': confidence,
            'filename': sample['filename'],
            'original_shape': (transform_params['original_height'], transform_params['original_width']),
            'img_path': image_path
        }
    
    def predict_batch(self, image_info_list, batch_size=BATCH_SIZE, use_amp=USE_AMP, save_dir=None):
        """
        Perform batch inference with comprehensive result saving and statistics
        
        Args:
            image_info_list: List of image info dictionaries
            batch_size: Batch processing size
            use_amp: Use Automatic Mixed Precision
            save_dir: Directory to save results
        
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        # Create dataset and dataloader
        dataset = XRayInferenceDataset(image_info_list,
                                     target_size=TARGET_SIZE,
                                     crop_margin=CROP_MARGIN)
        
        def custom_collate(batch):
            """Custom collate function for handling transform parameters and relative paths"""
            images = torch.stack([item['image'] for item in batch])
            img_paths = [item['img_path'] for item in batch]
            relative_paths = [item['relative_path'] for item in batch]
            filenames = [item['filename'] for item in batch]
            
            # Process transform parameters for batch
            transform_params_list = []
            for i in range(len(batch)):
                transform_params = {
                    'original_height': batch[i]['original_height'],
                    'original_width': batch[i]['original_width'],
                    'scale': batch[i]['scale'],
                    'new_height': batch[i]['new_height'],
                    'new_width': batch[i]['new_width'],
                    'pad_top': batch[i]['pad_top'],
                    'pad_bottom': batch[i]['pad_bottom'],
                    'pad_left': batch[i]['pad_left'],
                    'pad_right': batch[i]['pad_right']
                }
                transform_params_list.append(transform_params)
            
            return {
                'image': images,
                'img_path': img_paths,
                'relative_path': relative_paths,
                'filename': filenames,
                'transform_params': transform_params_list
            }
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True,
            collate_fn=custom_collate
        )
        
        results = []
        microstructure_data = []  # Store microstructure ratio data for analysis
        
        print(f"Starting inference on {len(image_info_list)} images")
        print(f"Processing configuration: crop_margin={CROP_MARGIN}, target_size={TARGET_SIZE}, resize_to_original={RESIZE_TO_ORIGINAL}")
        print(f"Microstructure analysis: {'Enabled (per experiment)' if OUTPUT_MICROSTRUCTURE_ANALYSIS else 'Disabled'}")
        print(f"AMP status: {'Enabled (CUDA)' if use_amp and self.device.type == 'cuda' else 'Disabled (CPU or manual override)'}")
        
        # Process batches
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch['image'].to(self.device)  # [B, 3, H, W]
            
            # Perform inference
            with torch.no_grad():
                if use_amp and self.device.type == 'cuda':
                    try:
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(images)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                else:
                    # CPU inference or AMP disabled
                    outputs = self.model(images)
            
            # Process outputs
            probabilities = torch.softmax(outputs, dim=1)  # [B, 2, H, W]
            predictions = torch.argmax(outputs, dim=1)     # [B, H, W]
            
            # Process each sample in the batch
            for i in range(images.shape[0]):
                try:
                    pred_mask = predictions[i].cpu().numpy()
                    prob_maps = probabilities[i].cpu().numpy()
                    filename = batch['filename'][i]
                    relative_path = batch['relative_path'][i]
                    transform_params = batch['transform_params'][i]
                    img_path = batch['img_path'][i]
                    confidence = prob_maps[1]  # Foreground confidence
                    
                    original_shape = (transform_params['original_height'], transform_params['original_width'])
                    
                    # Restore to original dimensions if requested
                    if RESIZE_TO_ORIGINAL:
                        pred_mask = inverse_transform(pred_mask, transform_params)
                        confidence = inverse_transform(confidence, transform_params)
                        # Process full probability maps
                        prob_maps_resized = np.zeros((2, transform_params['original_height'], transform_params['original_width']))
                        for j in range(2):
                            prob_maps_resized[j] = inverse_transform(prob_maps[j], transform_params)
                        prob_maps = prob_maps_resized
                    
                    result = {
                        'prediction': pred_mask,
                        'probabilities': prob_maps,
                        'confidence': confidence,
                        'filename': filename,
                        'relative_path': relative_path,
                        'original_shape': original_shape,
                        'img_path': img_path
                    }
                    
                    results.append(result)
                    
                    # Collect microstructure ratio data if enabled
                    if OUTPUT_MICROSTRUCTURE_ANALYSIS:
                        # mask_ratio = foreground pixels / total pixels
                        mask_ratio = np.sum(pred_mask == 1) / pred_mask.size
                        # microstructure_ratio = 1 - mask_ratio
                        microstructure_ratio = 1.0 - mask_ratio
                        
                        # Extract experiment name from relative path
                        experiment_name = Path(relative_path).parts[0] if Path(relative_path).parent != Path('.') else 'root'
                        frame_name = Path(filename).stem
                        
                        microstructure_data.append({
                            'experiment': experiment_name,
                            'frame': frame_name,
                            'microstructure_ratio': microstructure_ratio
                        })
                    
                    # Save results if directory specified
                    if save_dir:
                        self._save_single_result(result, save_dir)
                        
                except Exception as e:
                    print(f"Error processing sample {i} ({batch['filename'][i] if i < len(batch['filename']) else 'unknown'}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"Batch inference completed! Processed {len(results)} images.")
        
        # Calculate and save comprehensive statistics
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"\nInference Statistics:")
        print(f"Total images processed: {len(results)}")
        print(f"Total time: {inference_time:.2f}s")
        
        if len(results) > 0:
            print(f"Average time per image: {inference_time/len(results):.3f}s")
            print(f"Processing speed: {len(results)/inference_time:.1f} images/sec")
            
            # Calculate basic statistics
            foreground_ratios = []
            avg_confidences = []
            
            for result in results:
                pred_mask = result['prediction']
                confidence = result['confidence']
                
                fg_ratio = np.sum(pred_mask == 1) / pred_mask.size * 100
                avg_conf = np.mean(confidence[pred_mask == 1]) if np.any(pred_mask == 1) else 0
                
                foreground_ratios.append(fg_ratio)
                avg_confidences.append(avg_conf)
            
            print(f"Average foreground ratio: {np.mean(foreground_ratios):.2f}%")
            print(f"Average confidence: {np.mean(avg_confidences):.3f}")
            
            # Save statistics
            stats = {
                'total_images': len(results),
                'inference_time': inference_time,
                'avg_time_per_image': inference_time / len(results),
                'processing_speed': len(results) / inference_time,
                'avg_foreground_ratio': float(np.mean(foreground_ratios)),
                'avg_confidence': float(np.mean(avg_confidences)),
                'configuration': {
                    'model_path': MODEL_PATH,
                    'input_path': INPUT_PATH,
                    'architecture': MODEL_ARCHITECTURE,
                    'backbone': MODEL_BACKBONE,
                    'target_size': TARGET_SIZE,
                    'crop_margin': CROP_MARGIN,
                    'resize_to_original': RESIZE_TO_ORIGINAL,
                    'batch_size': BATCH_SIZE,
                    'use_amp': USE_AMP,
                    'device': DEVICE,
                    'output_probabilities': OUTPUT_PROBABILITIES,
                    'output_masks': OUTPUT_MASKS,
                    'output_overlay': OUTPUT_OVERLAY,
                    'output_microstructure_analysis': OUTPUT_MICROSTRUCTURE_ANALYSIS
                }
            }
            
            stats_path = Path(OUTPUT_DIR) / 'inference_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"Statistics saved to: {stats_path}")
            
            # Generate microstructure analysis if enabled and data available
            if OUTPUT_MICROSTRUCTURE_ANALYSIS and microstructure_data:
                self._save_microstructure_analysis_per_experiment(microstructure_data, OUTPUT_DIR)
            
            return True
        else:
            print("No images were successfully processed!")
            return False
    
    def _save_single_result(self, result, save_dir):
        """
        Save three independent output types: probability maps, binary masks, and overlay images
        Supports nested directory structure preservation
        
        Args:
            result: Inference result dictionary
            save_dir: Base directory for saving results
        """
        filename = Path(result['filename']).stem
        relative_path = Path(result['relative_path'])
        
        # Get subfolder path (e.g. experiment_A/)
        subfolder = relative_path.parent
        
        # Create three output directories for each subfolder
        subfolder_output = Path(save_dir) / subfolder
        prob_dir = subfolder_output / 'probabilities'
        mask_dir = subfolder_output / 'masks' 
        overlay_dir = subfolder_output / 'overlay'
        
        # Ensure directories exist
        prob_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Save probability heatmap (0.8-1.0 display range)
            if OUTPUT_PROBABILITIES:
                prob_path = prob_dir / f"{filename}_prob.png"
                confidence = result['confidence']
                
                # Create probability heatmap visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(confidence, cmap='hot', vmin=0.8, vmax=1.0)
                ax.set_title(f'Foreground Probability (0.8-1.0)\n{filename}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(prob_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # 2. Save binary segmentation mask (black/white)
            if OUTPUT_MASKS:
                mask_path = mask_dir / f"{filename}_mask.png"
                mask_img = (result['prediction'] * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_img)
            
            # 3. Save overlay image (original + red semi-transparent foreground)
            if OUTPUT_OVERLAY:
                overlay_path = overlay_dir / f"{filename}_overlay.png"
                
                # Load original image for overlay
                original_img = cv2.imread(result['img_path'], cv2.IMREAD_GRAYSCALE)
                if original_img is None:
                    print(f"Warning: Cannot load original image for overlay: {result['img_path']}")
                    return
                
                # Ensure size consistency
                if original_img.shape != result['prediction'].shape:
                    print(f"Warning: Size mismatch - original: {original_img.shape}, prediction: {result['prediction'].shape}")
                    return
                
                # Create overlay visualization
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
                
                # Create red foreground mask
                mask_colored = np.zeros_like(original_rgb)
                mask_colored[result['prediction'] == 1] = [0, 0, 255]  # Red in BGR format
                
                # Blend: original image + 30% transparent red mask
                alpha = 0.3
                overlay_img = cv2.addWeighted(original_rgb, 1-alpha, mask_colored, alpha, 0)
                
                cv2.imwrite(str(overlay_path), overlay_img)
            
        except Exception as e:
            print(f"Error saving results for {relative_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_microstructure_analysis_per_experiment(self, microstructure_data, save_dir):
        """
        Save microstructure ratio analysis per experiment as separate CSV files and curves
        
        Creates experiment-specific microstructure analysis:
        1. CSV file for each experiment with frame and microstructure_ratio columns
        2. Trend curves showing microstructure ratio evolution per experiment
        
        Args:
            microstructure_data: List of microstructure ratio data dictionaries
            save_dir: Base directory for saving analysis results
        """
        try:
            import pandas as pd
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(microstructure_data)
            
            print(f"\nGenerating microstructure analysis per experiment...")
            print(f"Total frames analyzed: {len(df)}")
            
            # Group by experiment and save separately
            experiments = df['experiment'].unique()
            print(f"Experiments found: {len(experiments)} ({', '.join(experiments)})")
            
            for experiment in experiments:
                exp_data = df[df['experiment'] == experiment].copy()
                exp_data = exp_data.sort_values('frame')  # Sort by frame name
                
                # Create experiment-specific directory
                exp_output_dir = Path(save_dir) / experiment
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save experiment-specific CSV (only frame and microstructure_ratio)
                csv_path = exp_output_dir / 'microstructure_analysis.csv'
                exp_data[['frame', 'microstructure_ratio']].to_csv(csv_path, index=False, float_format='%.6f')
                print(f"✅ CSV saved for {experiment}: {csv_path}")
                
                # Generate experiment-specific trend curve
                self._generate_single_experiment_curve(exp_data, exp_output_dir, experiment)
            
        except ImportError:
            print("Warning: pandas not available, saving basic CSV format per experiment")
            # Fallback to basic CSV without pandas
            experiments = {}
            
            # Group data by experiment
            for data in microstructure_data:
                exp_name = data['experiment']
                if exp_name not in experiments:
                    experiments[exp_name] = []
                experiments[exp_name].append(data)
            
            print(f"Experiments found: {len(experiments.keys())} ({', '.join(experiments.keys())})")
            
            # Save each experiment separately
            for exp_name, exp_data in experiments.items():
                exp_output_dir = Path(save_dir) / exp_name
                exp_output_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = exp_output_dir / 'microstructure_analysis.csv'
                with open(csv_path, 'w') as f:
                    f.write("frame,microstructure_ratio\n")
                    # Sort by frame name
                    sorted_data = sorted(exp_data, key=lambda x: x['frame'])
                    for data in sorted_data:
                        f.write(f"{data['frame']},{data['microstructure_ratio']:.6f}\n")
                
                print(f"✅ CSV saved for {exp_name}: {csv_path}")
                
                # Generate simple curve without pandas
                self._generate_simple_curve(sorted_data, exp_output_dir, exp_name)
            
        except Exception as e:
            print(f"Error generating microstructure analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_single_experiment_curve(self, exp_data, save_dir, experiment_name):
        """
        Generate microstructure ratio trend curve for a single experiment.
        X-axis uses integer indices with an automatic, nicely spaced locator.
        """
        try:
            # Sort by frame name and use 0..N-1 as the x-axis
            exp_data = exp_data.sort_values('frame').reset_index(drop=True)
            x = exp_data.index.to_numpy()
            y = exp_data['microstructure_ratio'].to_numpy()

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(x, y, 'o-', linewidth=3, markersize=6)

            ax.set_title(f"Microstructure Ratio - {experiment_name}", fontsize=18, fontweight='bold')
            ax.set_xlabel("Frame Index", fontsize=16, fontweight='bold')
            ax.set_ylabel("Microstructure Ratio", fontsize=16, fontweight='bold')

            # Cleaner grid and limits
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(1, len(x) - 1))

            # Key: show ~8 integer ticks across [0, N-1]
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            # Optional minor ticks every 1 frame (comment out if too dense)
            # ax.xaxis.set_minor_locator(MultipleLocator(1))

            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=14)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()
            curves_path = Path(save_dir) / "microstructure_curves.png"
            plt.savefig(curves_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Curve saved for {experiment_name}: {curves_path}")

        except Exception as e:
            print(f"Error generating microstructure curve for {experiment_name}: {e}")
            import traceback; traceback.print_exc()


    
    def _generate_simple_curve(self, exp_data, save_dir, experiment_name):
        """
        Generate simple microstructure curve without pandas (fallback)
        
        Args:
            exp_data: List of experiment data dictionaries
            save_dir: Directory to save curve plot
            experiment_name: Name of the experiment
        """
        try:
            # Create figure for single experiment
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            # Extract microstructure ratios and frame names
            ratios = [data['microstructure_ratio'] for data in exp_data]
            frames = [data['frame'] for data in exp_data]
            
            # Plot microstructure ratio trend
            ax.plot(range(len(ratios)), ratios, 'o-', color='blue', linewidth=3, markersize=6)
            
            ax.set_title(f'Microstructure Ratio - {experiment_name}', fontsize=18, fontweight='bold')
            ax.set_xlabel('Frame Index', fontsize=16, fontweight='bold')
            ax.set_ylabel('Microstructure Ratio', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Set bold font for axis tick labels
            ax.tick_params(axis='both', which='major', labelsize=14)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Set x-axis ticks to show frame names
            if len(frames) <= 20:
                ax.set_xticks(range(len(frames)))
                ax.set_xticklabels(frames, rotation=45, ha='right', fontsize=12, fontweight='bold')
            else:
                # Show every nth frame name for large datasets
                step = len(frames) // 10
                indices = range(0, len(frames), step)
                ax.set_xticks(indices)
                ax.set_xticklabels([frames[idx] for idx in indices], 
                                 rotation=45, ha='right', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the figure in experiment directory
            curves_path = Path(save_dir) / 'microstructure_curves.png'
            plt.savefig(curves_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Curve saved for {experiment_name}: {curves_path}")
            
        except Exception as e:
            print(f"Error generating simple microstructure curve for {experiment_name}: {e}")
            import traceback
            traceback.print_exc()

# ================================================================
# FILE DISCOVERY AND PATH UTILITIES
# ================================================================

def find_images_recursive(input_dir):
    """
    Recursively discover all image files in nested directory structures
    
    Args:
        input_dir: Parent directory containing experiment subdirectories
    
    Returns:
        image_info_list: List of dicts with 'path' and 'relative_path' keys
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    image_info_list = []
    
    # Recursively search for image files
    for ext in IMAGE_EXTENSIONS:
        for img_path in input_path.rglob(f'*{ext}'):
            relative_path = img_path.relative_to(input_path)
            image_info_list.append({
                'path': str(img_path),
                'relative_path': str(relative_path)
            })
    
    return sorted(image_info_list, key=lambda x: x['relative_path'])

def get_image_paths(input_path):
    """
    Get input image path list supporting both files and nested directory structures
    
    Args:
        input_path: Path to single file or directory
    
    Returns:
        image_info_list: List of image info dictionaries
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file processing
        return [{'path': str(input_path), 'relative_path': input_path.name}]
    elif input_path.is_dir():
        # Recursive directory structure processing
        image_info_list = find_images_recursive(input_path)
        
        if len(image_info_list) == 0:
            raise ValueError(f"No images found in {input_path} and its subdirectories")
        
        print(f"Found {len(image_info_list)} images in directory structure")
        
        # Display directory structure information
        folders = set(Path(info['relative_path']).parent for info in image_info_list)
        unique_folders = [f for f in folders if str(f) != '.']  # Exclude root directory
        if unique_folders:
            print(f"Processing {len(unique_folders)} subdirectories")
        
        return image_info_list
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

# ================================================================
# MAIN EXECUTION FUNCTION
# ================================================================

def main():
    """
    Main inference execution function
    
    Performs complete inference pipeline:
    1. Configuration validation
    2. Model loading
    3. Image discovery
    4. Batch inference
    5. Results saving and statistics
    """
    print("X-ray Segmentation Inference (Single File Edition - Per Experiment Analysis)")
    print("=" * 50)
    
    # Display configuration
    print("Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input: {INPUT_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Architecture: {MODEL_ARCHITECTURE} + {MODEL_BACKBONE}")
    print(f"  Target size: {TARGET_SIZE}")
    print(f"  Crop margin: {CROP_MARGIN}")
    print(f"  Resize to original: {RESIZE_TO_ORIGINAL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  AMP: {USE_AMP} ({'GPU only - will be disabled on CPU' if DEVICE == 'cpu' else 'Enabled'})")
    print(f"  Output probabilities: {OUTPUT_PROBABILITIES} (range: 0.8-1.0)")
    print(f"  Output masks: {OUTPUT_MASKS}")
    print(f"  Output overlay: {OUTPUT_OVERLAY}")
    print(f"  Output microstructure analysis: {OUTPUT_MICROSTRUCTURE_ANALYSIS} (per experiment)")
    print("=" * 50)
    
    # Validate required files
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH parameter to point to the correct model file")
        return
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input path not found: {INPUT_PATH}")
        print("Please update INPUT_PATH parameter to point to the correct input directory")
        return
    
    # Initialize inference pipeline
    try:
        inferencer = XRayInference(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Discover input images
    try:
        image_info_list = get_image_paths(INPUT_PATH)
        
        if len(image_info_list) == 0:
            print(f"⚠️ No images found with extensions: {IMAGE_EXTENSIONS}")
            return
        
        # Display file information
        print(f"Ready to process {len(image_info_list)} images")
        if len(image_info_list) <= 5:
            for info in image_info_list:
                print(f"    - {info['relative_path']}")
        elif len(image_info_list) <= 20:
            print("Sample files:")
            for info in image_info_list[:3]:
                print(f"    - {info['relative_path']}")
            print(f"    ... and {len(image_info_list) - 3} more files")
        else:
            print(f"Large batch: {len(image_info_list)} files across multiple directories")
        
    except Exception as e:
        print(f"❌ Error getting image paths: {e}")
        return
    
    # Execute inference
    print(f"\nStarting inference...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = inferencer.predict_batch(
        image_info_list=image_info_list,
        batch_size=BATCH_SIZE,
        use_amp=True,
        save_dir=OUTPUT_DIR
    )
    
    if success:
        print(f"\n✅ All done! Check results in: {OUTPUT_DIR}")
        print("Output structure (nested by experiment):")
        print(f"   experiment_A/")
        print(f"   ├── probabilities/  (probability heatmaps 0.8-1.0)")
        print(f"   ├── masks/          (black/white binary masks)")
        print(f"   ├── overlay/        (original images + red overlay)")
        if OUTPUT_MICROSTRUCTURE_ANALYSIS:
            print(f"   ├── microstructure_analysis.csv  (frame, microstructure_ratio)")
            print(f"   └── microstructure_curves.png    (trend visualization)")
        print(f"   experiment_B/")
        print(f"   ├── probabilities/")
        print(f"   ├── masks/")
        print(f"   ├── overlay/")
        if OUTPUT_MICROSTRUCTURE_ANALYSIS:
            print(f"   ├── microstructure_analysis.csv")
            print(f"   └── microstructure_curves.png")
        print("Successfully applied production-grade image processing pipeline!")
        print("Original directory structure preserved!")
        if OUTPUT_MICROSTRUCTURE_ANALYSIS:
            print("📊 Microstructure analysis generated per experiment:")
            print("   - Each experiment has its own CSV and curve files")
            print("   - CSV format: frame, microstructure_ratio")
            print("   - Curve shows trend for that experiment only")
    else:
        print("❌ Inference failed, please check error messages")
        return

if __name__ == "__main__":
    main()