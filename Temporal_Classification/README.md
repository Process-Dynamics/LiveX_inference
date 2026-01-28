# Temporal Classification for X-ray Microstructure Analysis

Multi-label temporal classification for X-ray microstructure images using Swin Transformer to identify six microstructure phases during metal solidification processes.

## Overview

This project implements automated classification of X-ray imaging data, capable of identifying the following six microstructure types:
- **Columnar**
- **Equiaxed**
- **Alpha (α-IMC)**
- **Liquid**
- **Beta (β-IMC)**
- **Hot_Tear**

The model uses a multi-label classification strategy with independent thresholds for each class, enabling simultaneous identification of multiple microstructures in a single image.

## Repository Structure

```
.
├── inference.py           # Inference script
├── environment.yml        # Conda environment configuration
└── README.md             # This file
```

**Note**: Model weights (`best_model.pth`) and test data (`test_data/`) are hosted separately. Download links are provided below.

---

## **Step 1: Download Required Files**

### 📥 Model Weights
**Download**: [Click here to download best_model.pth](https://drive.google.com/file/d/13-0IK0P3zKs9MuOmJx7s1h5YNQ63QHNx/view?usp=sharingD)

### 📥 Test Data
**Download**: [Click here to download test data](https://drive.google.com/file/d/1mNI6KBYijrUNcpSeYrc4khzlVp5sfdej/view?usp=sharing)

Download the `test_data` folder as example test data.

---

## **Step 2: Setup Environment**

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

---

## **Step 3: Configure Inference**

Open `inference.py` and modify the configuration at lines 453-462:

```python
# Path to trained model checkpoint
CHECKPOINT_PATH = r"/path/to/your/best_model.pth"

# Input directory (containing images or subdirectories)
INPUT_DIR = r"/path/to/your/test_data/"

# Output directory for results
OUTPUT_DIR = r"/path/to/output/results"

# Optional features
GENERATE_ANIMATION = True  # Generate MP4 animation
```

---

## **Step 4: Run Inference**

```bash
python inference.py
```

### Directory Mode Detection

The script automatically detects the input directory mode:
- **Single mode**: Input directory directly contains image files, processes a single experiment
- **Multi mode**: Input directory contains subdirectories, batch processes multiple experiments

---

## Output Files

After inference completion, the output directory will contain:

### 1. predictions.csv
Detailed prediction results for each frame:
- `frame_idx`: Frame index
- `filename`: Image filename
- `predicted_labels`: Active labels (separated by `|`)
- `num_active_labels`: Number of active labels
- `prob_ClassName`: Prediction probability for each class
- `pred_ClassName`: Binary prediction for each class (0 or 1)
- `threshold_ClassName`: Threshold used for each class

### 2. animation.mp4
Visualization animation video containing:
- Left: Original image sequence
- Right: Probability curves for each class over time
- Text box: Currently active labels and highest probability class

---

## Model Configuration

The model uses the following configuration:
- **Architecture**: Swin Transformer Large (swin_large_patch4_window12_384_in22k)
- **Input Size**: 1056×1056
- **Number of Classes**: 6
- **Preprocessing**: ImageNet normalization, longest-edge scaling with zero padding

### Per-Class Thresholds

Each class uses independently optimized thresholds:
```python
CLASS_THRESHOLDS = {
    0: 0.35,  # Columnar
    1: 0.60,  # Equiaxed
    2: 0.20,  # Alpha (α-IMC)
    3: 0.50,  # Liquid
    4: 0.65,  # Beta (β-IMC)
    5: 0.25   # Hot_Tear
}
```

---

## Notes

1. Ensure GPU is available for optimal performance (script automatically detects and uses GPU)
2. Supported image formats: PNG, JPG, JPEG (case-insensitive)
3. Animation generation requires FFmpeg installed on the system
4. Model uses longest-edge scaling strategy to preserve original aspect ratio

---

## Troubleshooting

**Q: Model file not found error**  
A: Check that `CHECKPOINT_PATH` correctly points to the `best_model.pth` file

**Q: Cannot generate animation**  
A: Ensure FFmpeg is installed, or set `GENERATE_ANIMATION` to `False`

**Q: GPU out of memory**  
A: Model will automatically fall back to CPU, but inference will be slower

---

## Citation

If you use this project, please cite the relevant paper or data source.

---

## Contact

For questions, please contact: [To be added]
