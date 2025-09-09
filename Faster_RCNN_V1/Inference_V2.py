import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# Set the device, prioritizing GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# ----------------------------------------
# 1. Utility Functions
# ----------------------------------------

def natural_sort_key(s):
    """
    Sorting key for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# ----------------------------------------
# 2. Model Construction
# ----------------------------------------

def create_faster_rcnn_model_50_V2(num_classes):
    """
    Create a Faster RCNN model with ResNet50 + FPN as the backbone (V1 version)
    """
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ----------------------------------------
# 3. Inference Functions
# ----------------------------------------

def perform_inference(model, image_tensor):
    """
    Perform inference on a given image tensor using the trained model.
    """
    with torch.no_grad():
        predictions = model([image_tensor])
    return {
        "boxes": predictions[0]['boxes'].cpu().numpy(),
        "scores": predictions[0]['scores'].cpu().numpy(),
        "labels": predictions[0]['labels'].cpu().numpy()
    }

def visualize_and_save_predictions(image, boxes, scores, output_image_path, score_threshold=0.5):
    """
    Visualize detection results and save the annotated image.
    """
    image_np = np.array(image)
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score >= score_threshold:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
            cv2.putText(image_np, f'{score:.2f}', (xmin, ymin - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imwrite(output_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

# ----------------------------------------
# 4. Image Processing Functions
# ----------------------------------------

def process_image(model, image, image_filename, output_image_folder, frame_idx, 
                  bbox_counts=None, score_threshold=0.5, nms_iou_threshold=0.3, 
                  plot_output_images=True, count_bboxes=True):
    """
    Process a single image for inference and optionally count bounding boxes.
    """
    # Perform model inference
    predictions = perform_inference(model, F.to_tensor(image).to(device))
    
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    
    # Convert to tensors if needed
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes).to(device)
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores).to(device)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels).to(device)
    
    # Filter by score threshold
    high_score_indices = scores >= score_threshold
    boxes = boxes[high_score_indices]
    scores = scores[high_score_indices]
    labels = labels[high_score_indices]
    
    # Apply NMS
    if len(boxes) > 0:
        nms_indices = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]
        labels = labels[nms_indices]
    
    # Convert back to numpy for processing
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    bbox_count = len(boxes_np)
    
    # Count bounding boxes if enabled
    if count_bboxes and bbox_counts is not None:
        bbox_counts.append({
            'frame': frame_idx,
            'filename': image_filename,
            'bbox_count': bbox_count
        })
    
    # Save visualization if enabled
    if plot_output_images and output_image_folder:
        output_image_path = os.path.join(output_image_folder, f"output_{image_filename}")
        visualize_and_save_predictions(image, boxes_np, scores_np, 
                                     output_image_path, score_threshold)
    
    return bbox_count

def process_images_in_folder(model, folder_path, result_folder,
                           score_threshold=0.5, nms_iou_threshold=0.3, 
                           plot_output_images=True, count_bboxes=True):
    """
    Perform inference on all images in a folder and optionally count bounding boxes.
    """
    # Setup output folders
    if plot_output_images:
        output_image_folder = os.path.join(result_folder, 'Output_Img')
        os.makedirs(output_image_folder, exist_ok=True)
    else:
        output_image_folder = None
    
    os.makedirs(result_folder, exist_ok=True)
    
    # Get all image files and sort them naturally
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
    image_files = sorted(image_files, key=natural_sort_key)
    
    bbox_counts = [] if count_bboxes else None
    
    for frame_idx, filename in enumerate(tqdm(image_files, desc="Processing images", unit="img")):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("RGB")
        
        bbox_count = process_image(
            model, image, filename, output_image_folder, frame_idx,
            bbox_counts, score_threshold, nms_iou_threshold, 
            plot_output_images, count_bboxes
        )
    
    # Save bbox count statistics if enabled
    if count_bboxes and bbox_counts:
        bbox_df = pd.DataFrame(bbox_counts)
        csv_path = os.path.join(result_folder, 'bbox_counts.csv')
        bbox_df.to_csv(csv_path, index=False)
        print(f"Bbox counts saved to: {csv_path}")
        
        # Generate statistics plot
        generate_bbox_statistics_plot(bbox_df, result_folder)
        
        # Print summary statistics
        print(f"\nDetection Summary:")
        print(f"Total frames: {len(bbox_counts)}")
        print(f"Total bboxes detected: {bbox_df['bbox_count'].sum()}")
        print(f"Average bboxes per frame: {bbox_df['bbox_count'].mean():.2f}")
        print(f"Max bboxes in single frame: {bbox_df['bbox_count'].max()}")
        print(f"Min bboxes in single frame: {bbox_df['bbox_count'].min()}")
        
        return bbox_df
    else:
        return None

def generate_bbox_statistics_plot(bbox_df, result_folder):
    """
    Generate plots showing bbox count statistics over frames.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Bbox count over frames
    ax1.plot(bbox_df['frame'], bbox_df['bbox_count'], 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Number of Bounding Boxes')
    ax1.set_title('Bounding Box Count Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(bbox_df['frame'], bbox_df['bbox_count'], 1)
    p = np.poly1d(z)
    ax1.plot(bbox_df['frame'], p(bbox_df['frame']), "r--", alpha=0.8, linewidth=2, label=f'Trend')
    ax1.legend()
    
    # Plot 2: Histogram of bbox counts
    ax2.hist(bbox_df['bbox_count'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Number of Bounding Boxes')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Bounding Box Counts')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(result_folder, 'bbox_statistics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Statistics plot saved to: {plot_path}")

# ----------------------------------------
# 5. Video Creation Function
# ----------------------------------------

def create_video_from_images(image_folder_path, experiment_name, output_folder, fps=10):
    """
    Create a video from images in the specified folder.
    """
    video_path = os.path.join(output_folder, f"{experiment_name}.mp4")
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    frames = [f for f in os.listdir(image_folder_path) 
              if f.lower().endswith(image_extensions)]
    frames = sorted(frames, key=natural_sort_key)
    
    if not frames:
        print("No images found for video creation.")
        return False
    
    # Read first image to get dimensions
    first_image_path = os.path.join(image_folder_path, frames[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Cannot read the first image: {first_image_path}")
        return False
    
    img_size = (first_image.shape[1], first_image.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    
    frame_count = 0
    for frame in frames:
        f_path = os.path.join(image_folder_path, frame)
        image = cv2.imread(f_path)
        if image is None:
            continue
        
        # Resize if necessary
        if (image.shape[1], image.shape[0]) != img_size:
            image = cv2.resize(image, img_size)
        
        videowriter.write(image)
        frame_count += 1
    
    videowriter.release()
    print(f"Video saved to {video_path}, total frames: {frame_count}")
    return True

# ----------------------------------------
# 6. Main Function
# ----------------------------------------

def main(num_classes, folder_path, model_weights_path, result_folder_path,
         score_threshold=0.5, nms_iou_threshold=0.5, 
         plot_output_images=True, count_bboxes=True, create_video=True):
    """
    Main function: load the model, process images, optionally count bboxes, and generate statistics.
    """
    print("Loading model...")
    
    # Load the model
    model = create_faster_rcnn_model_50_V2(num_classes)
    
    # Load state dict directly (not wrapped in a dictionary)
    state_dict = torch.load(model_weights_path, map_location=device)
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Process images and generate statistics
    bbox_df = process_images_in_folder(
        model, folder_path, result_folder_path,
        score_threshold, nms_iou_threshold, plot_output_images, count_bboxes
    )
    
    # Create video if enabled
    if plot_output_images and create_video:
        output_image_folder = os.path.join(result_folder_path, "Output_Img")
        if os.path.exists(output_image_folder):
            create_video_from_images(output_image_folder, 'Detection_Results', 
                                   result_folder_path, fps=10)

# ----------------------------------------
# Entry Point
# ----------------------------------------

if __name__ == "__main__":
    # Configuration parameters
    num_classes = 2  # Set the number of classes in the dataset
    FOLDER_PATH = r"C:\Users\lina4366\Desktop\Faster_RCNN\Faster_RCNN__Evaluation\Faster_RCNN__Evaluation\test_demo"
    MODEL_WEIGHTS_PATH = r"C:\Users\lina4366\Desktop\Faster_RCNN\Faster_RCNN__Evaluation\Faster_RCNN__Evaluation\model_V2_weight.pth"
    RESULT_FOLDER_PATH = r"C:\Users\lina4366\Desktop\Faster_RCNN\Faster_RCNN__Evaluation\Faster_RCNN__Evaluation\Result"
    
    # Inference thresholds
    score_threshold = 0.5        # Confidence threshold
    nms_iou_threshold = 0.3     # NMS threshold
    
    # Output options
    plot_output_images = True    # Set to False to skip image visualization  
    count_bboxes = True          # Set to False to skip bbox counting and statistics
    create_video = True          # Set to False to skip video creation
    
    # Run the main function
    main(num_classes, FOLDER_PATH, MODEL_WEIGHTS_PATH, RESULT_FOLDER_PATH,
         score_threshold, nms_iou_threshold, plot_output_images, count_bboxes, create_video)
