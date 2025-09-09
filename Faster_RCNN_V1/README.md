# Faster R-CNN Inference

This repository provides an inference pipeline for object detection
using **Faster R-CNN with ResNet50 + FPN** backbone. The script
`Inference_V1.py` supports batch processing of images, visualization of
results, bounding box statistics, and optional video creation.

------------------------------------------------------------------------

## 1. Environment Setup

To create the environment from the provided `environment.yml` file:

``` bash
conda env create -f environment.yml
conda activate faster_rcnn_env
```

------------------------------------------------------------------------

## 2. Model Weights

The trained model weights can be downloaded from Google Drive:

[📥 Download Model V1
Weights](https://drive.google.com/file/d/1Swm7vXu2T3pnv-6zHpwlg8esK4wXf1Hz/view?usp=drive_link)

[📥 Download Model V2
Weights]([https://drive.google.com/file/d/1Swm7vXu2T3pnv-6zHpwlg8esK4wXf1Hz/view?usp=drive_link](https://drive.google.com/file/d/1VrE0w32d0mxC6B2UD5NOsk6m-rdK4Y2r/view?usp=drive_link)

Save the file as `model_xxx.pth` and update the path in
`Inference_V1.py` accordingly.

------------------------------------------------------------------------

## 3. Running Inference

Run the inference script with:

``` bash
python Inference_V1.py
```

The main configuration parameters are defined at the bottom of the
script (`__main__` section):

``` python
num_classes = 2  # Number of classes in the dataset
FOLDER_PATH = r"path/to/input/images"
MODEL_WEIGHTS_PATH = r"path/to/model_weights.pth"
RESULT_FOLDER_PATH = r"path/to/save/results"

score_threshold = 0.5       # Confidence threshold for detections
nms_iou_threshold = 0.5     # IoU threshold for Non-Maximum Suppression
plot_output_images = True   # Save annotated images
count_bboxes = True         # Save bbox counts & statistics
create_video = True         # Create a video from annotated images
```

------------------------------------------------------------------------

## 4. Output Description

-   **Annotated Images**: Saved under `Result/Output_Img/` (if
    `plot_output_images=True`).\
-   **Bounding Box Counts**: CSV file `bbox_counts.csv` with per-frame
    statistics (if `count_bboxes=True`).\
-   **Statistics Plot**: `bbox_statistics.png` showing bbox count
    trends.\
-   **Summary Statistics**: Printed in console (total detections,
    average per frame, etc.).\
-   **Video**: `Detection_Results.mp4` created from annotated images (if
    `create_video=True`).

------------------------------------------------------------------------

## 5. Inference Options

  --------------------------------------------------------------------------------------
  Parameter              Type    Default   Description
  ---------------------- ------- --------- ---------------------------------------------
  `num_classes`          int     2         Number of classes (including background).

  `FOLDER_PATH`          str     ---       Path to input folder containing images.

  `MODEL_WEIGHTS_PATH`   str     ---       Path to `.pth` model weights file.

  `RESULT_FOLDER_PATH`   str     ---       Path to save results (annotated images, CSV,
                                           plots, video).

  `score_threshold`      float   0.5       Confidence threshold for keeping detections.

  `nms_iou_threshold`    float   0.5       IoU threshold for Non-Maximum Suppression.

  `plot_output_images`   bool    True      Save annotated detection results as images.

  `count_bboxes`         bool    True      Save bbox counts (CSV + plots).

  `create_video`         bool    True      Save a video (`.mp4`) compiled from annotated
                                           images.
  --------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Example Workflow

1.  Place your test images in a folder, e.g. `./test_demo/`.\
2.  Download the trained model weights from the link above and place
    them in your project directory.\
3.  Modify the paths in `Inference_V1.py` accordingly.\
4.  Run the script:

``` bash
python Inference_V1.py
```

Results will be saved under the `Result/` folder.
