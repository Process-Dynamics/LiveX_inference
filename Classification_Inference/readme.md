# Microstructure Classification Inference

Deep learning model inference tool for microstructure image classification (Equiaxed, Columnar, Background)

---

## Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate Classification_inference

```




---
## Download Model & Test_images.

### Download Model
[📥 Download Model
Weights](https://drive.google.com/drive/folders/1m38p9D4gXDtqj86etUUCENpdzr-eg9Q0?usp=drive_link)

### Download Images
[📥 Download Test Image 
Folder](https://drive.google.com/drive/folders/1W4_2om5Blme8Q2HrIB5lBU_XB6FZwuXb?usp=drive_link)

---
## Usage
### 1. Configure Parameters

Modify the configuration in `Inference.py`:

```python
INPUT_PATH = '/path/to/your/data'           # Input path
RESULT_FOLDER = '/path/to/results'          # Result output path
MODEL_WEIGHTS_PATH = '/path/to/model.pth'   # Model weights path
model_type = "swin"                         # Model type: "swin" or "resnet"
generate_animations = True                   # Whether to generate animations
```

### 2. Run Inference

```bash
python Inference.py
```

---

## Supported Input Formats

### Single Image File
```
input.jpg
```

### Folder Containing Images
```
my_images/
├── image1.png
├── image2.jpg
└── image3.png
```

### Multiple Experiment Folders
```
experiments/
├── exp1/
│   ├── img1.png
│   └── img2.png
└── exp2/
    └── img3.png
```

---

## Output Results

### File Structure
```
results/
└── folder_name/
    ├── inference_results.csv    # Prediction results table
    ├── probability_curves.png   # Probability curve plot
    └── animation.mp4           # Animation (optional)
```

### Detailed Description

#### inference_results.csv
Contains detailed prediction results for each image:
- `Filename` - Image file path
- `Predicted_Label` - Predicted class number (0/1/2)
- `Predicted_Name` - Predicted class name (Equiax/Columnar/Background)
- `Prob_Class_0` - Equiaxed probability
- `Prob_Class_1` - Columnar probability
- `Prob_Class_2` - Background probability
- `Confidence` - Prediction confidence

#### probability_curves.png
Shows the probability curves of each class over time for the entire sequence:
- Blue line: Equiaxed probability
- Red line: Columnar probability
- Green line: Background probability

#### animation.mp4 
Dynamic visualization of the inference process (requires `generate_animations=True`):
- Left side: Original image sequence
- Right side: Real-time probability curve updates

---

## File Description

- `Inference.py` - Main inference script
- `Model.py` - Model definitions
- `utils_analysis.py` - Result analysis and animation generation
- `environment.yml` - Conda environment configuration
