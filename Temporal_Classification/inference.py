"""
Standalone Inference Script for Multi-label X-ray Microstructure Classification

This script is self-contained and does not require external config or model files.
Dependencies: torch, timm, torchvision, PIL, numpy, pandas, matplotlib, tqdm

Usage:
    1. Modify the CONFIGURATION section below
    2. Run: python inference_standalone.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import glob
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from torchvision import transforms


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class InferenceConfig:
    """All configuration parameters for inference"""
    
    # ----- Model Parameters -----
    MODEL_NAME = 'swin_large_patch4_window12_384_in22k'
    NUM_CLASSES = 6
    IMG_SIZE = 1056
    IN_CHANNELS = 3
    
    # ----- Class Definitions -----
    CLASS_NAMES = {
        0: "Columnar",
        1: "Equiaxed",
        2: "Alpha",
        3: "Liquid",
        4: "Beta",
        5: "Hot_Tear"
    }
    
    # ----- Per-class Thresholds for Multi-label Classification -----
    CLASS_THRESHOLDS = {
        0: 0.35,  # Columnar
        1: 0.60,  # Equiaxed
        2: 0.20,  # Alpha
        3: 0.50,  # Liquid
        4: 0.65,  # Beta
        5: 0.25   # Hot_Tear
    }
    
    # ----- Normalization (ImageNet) -----
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# MODEL BUILDING
# ==============================================================================

def build_model(model_name, num_classes, img_size, in_channels, pretrained=False):
    """
    Build Swin Transformer model using timm library.
    
    Args:
        model_name: Name of the model in timm library
        num_classes: Number of output classes
        img_size: Input image size
        in_channels: Number of input channels
        pretrained: Whether to load pretrained weights (False for inference)
    
    Returns:
        Swin Transformer model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        in_chans=in_channels
    )
    return model


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
    
    Returns:
        Loaded model and epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if trained with DDP
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("Removed 'module.' prefix from DDP checkpoint")
    
    model.load_state_dict(state_dict)
    epoch = checkpoint.get('epoch', 'unknown')
    
    return model, epoch


# ==============================================================================
# IMAGE PREPROCESSING
# ==============================================================================

def preprocess_image(image_path, img_size, normalize_mean, normalize_std, device):
    """
    Preprocess image for inference.
    Uses longest-edge scaling with zero padding.
    
    Args:
        image_path: Path to image file
        img_size: Target image size
        normalize_mean: Normalization mean values
        normalize_std: Normalization std values
        device: Device to load tensor to
    
    Returns:
        Preprocessed tensor [1, C, H, W] and original PIL image
    """
    img = Image.open(image_path).convert('RGB')
    orig_img = img.copy()
    
    # Get original size
    w, h = img.size
    
    # Scale by longest edge
    if max(w, h) != img_size:
        if w > h:
            new_w = img_size
            new_h = int(h * img_size / w)
        else:
            new_h = img_size
            new_w = int(w * img_size / h)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Calculate zero padding to make square
    w, h = img.size
    pad_w = (img_size - w) // 2
    pad_h = (img_size - h) // 2
    padding = (pad_w, pad_h, img_size - w - pad_w, img_size - h - pad_h)
    
    # Apply zero padding
    img = transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
    
    # Convert to tensor and normalize
    img_tensor = transforms.functional.to_tensor(img)
    img_tensor = transforms.functional.normalize(img_tensor, mean=normalize_mean, std=normalize_std)
    
    return img_tensor.unsqueeze(0).to(device), orig_img


# ==============================================================================
# INFERENCE ENGINE
# ==============================================================================

class InferenceEngine:
    """Main inference engine for multi-label classification"""
    
    def __init__(self, checkpoint_path, config=None, generate_animation=True):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: InferenceConfig instance (uses default if None)
            generate_animation: Whether to generate animation video
        """
        self.config = config or InferenceConfig()
        self.generate_animation = generate_animation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        print(f"Loading model from: {checkpoint_path}")
        
        # Build and load model
        self.model = build_model(
            model_name=self.config.MODEL_NAME,
            num_classes=self.config.NUM_CLASSES,
            img_size=self.config.IMG_SIZE,
            in_channels=self.config.IN_CHANNELS,
            pretrained=False
        ).to(self.device)
        
        self.model, epoch = load_checkpoint(self.model, checkpoint_path, self.device)
        print(f"Model loaded successfully from epoch {epoch}")
        
        # Print per-class thresholds
        print("\nPer-class thresholds:")
        for class_idx, class_name in self.config.CLASS_NAMES.items():
            threshold = self.config.CLASS_THRESHOLDS[class_idx]
            print(f"  {class_name}: {threshold}")
        
        self.model.eval()
    
    def predict(self, image_path):
        """
        Predict single image (multi-label with per-class thresholds).
        
        Args:
            image_path: Path to image
        
        Returns:
            Dictionary with prediction results
        """
        img_tensor, orig_img = preprocess_image(
            image_path,
            self.config.IMG_SIZE,
            self.config.NORMALIZE_MEAN,
            self.config.NORMALIZE_STD,
            self.device
        )
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
            # Apply per-class thresholds
            predicted_labels = np.zeros(len(self.config.CLASS_NAMES), dtype=int)
            for class_idx in range(len(self.config.CLASS_NAMES)):
                threshold = self.config.CLASS_THRESHOLDS[class_idx]
                predicted_labels[class_idx] = int(probs[class_idx] > threshold)
            
            active_classes = [i for i, active in enumerate(predicted_labels) if active]
        
        result = {
            'probabilities': probs,
            'predicted_labels': predicted_labels,
            'active_classes': active_classes,
            'active_class_names': [self.config.CLASS_NAMES[i] for i in active_classes],
            'original_image': orig_img
        }
        
        return result
    
    def detect_directory_mode(self, directory):
        """
        Auto-detect whether directory contains images or subdirectories.
        
        Returns:
            'single' if directory contains images directly
            'multi' if directory contains subdirectories
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        has_images = any(glob.glob(os.path.join(directory, ext)) for ext in image_extensions)
        
        if has_images:
            return 'single'
        
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if subdirs:
            return 'multi'
        
        raise ValueError(f"Directory '{directory}' contains neither images nor subdirectories")
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a single directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images in {input_dir}")
        
        # Process each image
        results = []
        all_images = []
        
        for idx, img_path in enumerate(tqdm(image_files, desc="Inference")):
            result = self.predict(img_path)
            
            # Build result row
            row = {
                'frame_idx': idx,
                'filename': os.path.basename(img_path),
                'predicted_labels': '|'.join(result['active_class_names']) if result['active_class_names'] else 'None',
                'num_active_labels': len(result['active_classes'])
            }
            
            # Add per-class probabilities, predictions, and thresholds
            for class_idx, class_name in self.config.CLASS_NAMES.items():
                row[f'prob_{class_name}'] = result['probabilities'][class_idx]
                row[f'pred_{class_name}'] = result['predicted_labels'][class_idx]
                row[f'threshold_{class_name}'] = self.config.CLASS_THRESHOLDS[class_idx]
            
            results.append(row)
            all_images.append(result['original_image'])
        
        # Save CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved predictions to {csv_path}")
        
        # Generate animation
        if self.generate_animation:
            animation_path = os.path.join(output_dir, 'animation.mp4')
            self._generate_animation(all_images, df, animation_path)
            print(f"Saved animation to {animation_path}")
    
    def process_multiple_directories(self, root_dir, output_base_dir):
        """
        Process multiple experiment directories.
        
        Args:
            root_dir: Root directory containing experiment subdirectories
            output_base_dir: Base directory for outputs
        """
        experiment_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        print(f"Found {len(experiment_dirs)} experiments")
        
        for exp_dir in experiment_dirs:
            exp_name = os.path.basename(exp_dir)
            output_dir = os.path.join(output_base_dir, exp_name)
            
            print(f"\n{'='*60}")
            print(f"Processing: {exp_name}")
            print(f"{'='*60}")
            
            self.process_directory(exp_dir, output_dir)
    
    def _generate_animation(self, images, results_df, save_path):
        """Generate animation with image and probability curves (with per-class threshold markers)"""
        num_frames = len(images)
        
        # Setup plot style
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.weight'] = 'bold'
        
        fig, (ax_img, ax_prob) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Image subplot
        img_display = ax_img.imshow(np.array(images[0]))
        ax_img.axis('off')
        ax_img.set_title('Frame Image', fontsize=16, fontweight='bold')
        
        # Probability subplot
        lines = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.config.CLASS_NAMES)))
        
        for class_idx, class_name in self.config.CLASS_NAMES.items():
            color = colors[class_idx]
            
            # Probability line (threshold shown in legend only)
            line, = ax_prob.plot([], [], 'o-', linewidth=2.5, markersize=6,
                                label=f'{class_name} (T={self.config.CLASS_THRESHOLDS[class_idx]:.2f})',
                                color=color)
            lines[class_idx] = line
        
        ax_prob.set_xlim(0, num_frames - 1)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_xlabel('Frame Index', fontsize=14, fontweight='bold')
        ax_prob.set_ylabel('Probability', fontsize=14, fontweight='bold')
        ax_prob.set_title('Class Probabilities (Per-class Thresholds)', fontsize=16, fontweight='bold')
        ax_prob.legend(loc='upper right', fontsize=8)
        ax_prob.grid(True, alpha=0.3)
        
        # Current prediction text box
        label_text = ax_prob.text(0.02, 0.98, '', transform=ax_prob.transAxes,
                                  fontsize=12, fontweight='bold', verticalalignment='top',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                                           alpha=0.8, edgecolor='black', linewidth=2))
        
        plt.tight_layout()
        
        # Animation data
        x_data = []
        prob_history = {k: [] for k in self.config.CLASS_NAMES.keys()}
        
        def update(frame_idx):
            img_display.set_data(np.array(images[frame_idx]))
            
            current_labels = results_df.loc[frame_idx, 'predicted_labels']
            num_active = results_df.loc[frame_idx, 'num_active_labels']
            
            # Find max probability class
            max_prob = 0
            max_class = None
            for class_idx, class_name in self.config.CLASS_NAMES.items():
                prob = results_df.loc[frame_idx, f'prob_{class_name}']
                if prob > max_prob:
                    max_prob = prob
                    max_class = class_name
            
            label_text.set_text(f'Active ({num_active}):\n{current_labels}\n\nMax: {max_class}\n({max_prob:.3f})')
            
            x_data.append(frame_idx)
            for class_idx, class_name in self.config.CLASS_NAMES.items():
                prob = results_df.loc[frame_idx, f'prob_{class_name}']
                prob_history[class_idx].append(prob)
                lines[class_idx].set_data(x_data, prob_history[class_idx])
            
            return img_display, label_text, *lines.values()
        
        anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
        writer = FFMpegWriter(fps=10, bitrate=2000)
        anim.save(save_path, writer=writer)
        plt.close()
        
        plt.rcParams.update(plt.rcParamsDefault)


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main entry point for inference.
    Modify the settings below to customize behavior.
    """
    
    # ==================== USER SETTINGS ====================
    
    # Path to trained model checkpoint
    CHECKPOINT_PATH = r"/home/shun/Project/S3/Temporal_Classification/inference_package/best_model.pth"
    
    # Input directory (containing images or subdirectories)
    INPUT_DIR = r"/home/shun/Project/S3/Temporal_Classification/Test_data2/0096"
    
    # Output directory for results
    OUTPUT_DIR = r"/home/shun/Project/S3/Temporal_Classification/inference_package/inference_test/0119"
    
    # Optional features
    GENERATE_ANIMATION = True  # Generate MP4 animation
    
    # ==================================================
    
    # Create inference engine
    engine = InferenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        generate_animation=GENERATE_ANIMATION
    )
    
    # Auto-detect directory mode and run inference
    mode = engine.detect_directory_mode(INPUT_DIR)
    print(f"\nDetected mode: {mode.upper()}")
    print(f"{'='*60}\n")
    
    if mode == 'single':
        engine.process_directory(INPUT_DIR, OUTPUT_DIR)
    else:
        engine.process_multiple_directories(INPUT_DIR, OUTPUT_DIR)
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()