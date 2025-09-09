# X-ray Image Segmentation System

Metal solidification liquid phase segmentation inference pipeline based on UNet++ and EfficientNet-B6 architecture, supporting batch processing and liquid phase ratio analysis.

## System Requirements

- NVIDIA GPU (CUDA 12.1+ support)
- Python 3.10
- Conda package manager

## Installation Guide

### Create Conda Environment
```bash
# Create environment using the provided environment file
conda env create -f environment.yml

# Activate environment
conda activate xray_segmentation
```

### Verify Installation
```python
import torch
import segmentation_models_pytorch as smp
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Download Model and Test Data

### Model File
[📥 Download Model
Weights](https://drive.google.com/file/d/1EOSEyY2xvzZKMjvZ1kIH6SePHBQWcDOU/view?usp=drive_link)

### Test Images
[📥 Download Test Image 
Folder](https://drive.google.com/drive/folders/1vCsBLM4t3umcqe1AxLDB42kFBTZmZZju?usp=drive_link)

## Usage

### 1. Prepare Data
Organize your X-ray images in the following structure:
```
Test_images/
├── experiment_A/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── experiment_B/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── experiment_C/
    ├── image001.png
    └── ...
```

### 2. Configure Parameters
Edit the configuration section in `inference.py`:

```python
# Required paths
MODEL_PATH = r"path/to/your/best_model.pth"
INPUT_PATH = r"path/to/your/Test_images"
OUTPUT_DIR = r"path/to/your/Result"

# Adjustable parameters
BATCH_SIZE = 1                           # Batch processing size
DEVICE = "cuda"                          # "cuda" or "cpu"
```

### Feature Switches
```python
USE_AMP = False                         # Automatic Mixed Precision
OUTPUT_PROBABILITIES = True             # Generate probability maps
OUTPUT_MASKS = True                     # Generate binary masks
OUTPUT_OVERLAY = True                   # Generate overlay images
OUTPUT_MICROSTRUCTURE_ANALYSIS = True   # Liquid phase ratio analysis
```

### 3. Run Inference
```bash
python inference.py
```

## Output Structure

After inference completion, the following structure will be generated in the output directory:

```
Result/
├── experiment_A/
│   ├── probabilities/           # Probability heatmaps (0.8-1.0 range)
│   │   ├── image001_prob.png
│   │   └── image002_prob.png
│   ├── masks/                   # Binary segmentation masks (black/white)
│   │   ├── image001_mask.png
│   │   └── image002_mask.png
│   ├── overlay/                 # Original image + red foreground overlay
│   │   ├── image001_overlay.png
│   │   └── image002_overlay.png
│   ├── microstructure_analysis.csv    # Liquid phase ratio data
│   └── microstructure_curves.png      # Trend visualization
├── experiment_B/
│   └── (same structure)
└── inference_stats.json         # Inference statistics
```

### Output File Description

#### 1. Probability Maps (probabilities/)
- Heatmaps showing foreground probability
- Range: 0.8-1.0 (high confidence regions)
- Color mapping: Heatmap (black=low probability, red=high probability)

#### 2. Segmentation Masks (masks/)
- Binary segmentation results
- Black (0): Background
- White (255): Foreground/liquid phase regions

#### 3. Overlay Images (overlay/)
- Original image + 30% transparent red foreground
- Convenient for intuitive segmentation quality assessment

#### 4. Liquid Phase Ratio Analysis
- **CSV file**: Contains frame and microstructure_ratio columns
- **Trend curves**: Shows liquid phase ratio changes within experiments
- **Calculation formula**: microstructure_ratio = 1 - (foreground_pixels/total_pixels)
