import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import seaborn as sns


def create_animation_for_subfolder(subfolder_path, save_animation_path, orig_images_base_dir, with_heatmap=False):
    """
    For a given subfolder, read the prediction results and class probabilities from inference_results.csv,
    and combine the corresponding original images and (optional) CAM heatmaps to generate an animation (.mp4).
    If with_heatmap is True, the animation will include the original image, CAM heatmap, and probability curves;
    otherwise, it will only include the original image and probability curves.
    """
    print(f"Processing folder: {subfolder_path}")
    
    # Check if the CSV file exists
    csv_path = os.path.join(subfolder_path, "inference_results.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}, skipping this folder.")
        return

    # Read the CSV file
    results_df = pd.read_csv(csv_path)
    
    # Check if required columns exist, if Prob_Class_3 column is missing, create a zero-filled column
    required_columns = [
        'Filename', 'Predicted_Label',
        'Prob_Class_0', 'Prob_Class_1', 'Prob_Class_2'
    ]
    if not all(col in results_df.columns for col in required_columns):
        print(f"Missing required columns in {csv_path}, cannot generate animation. Skipping this folder.")
        return
    
    # If the fourth class probability column is missing, add a zero-filled column
    if 'Prob_Class_3' not in results_df.columns:
        results_df['Prob_Class_3'] = 0.0
        print("Added Prob_Class_3 column (all zeros) to be compatible with four-class animation generation.")

    # Get class probabilities
    prob_class_0 = results_df['Prob_Class_0'].tolist()
    prob_class_1 = results_df['Prob_Class_1'].tolist()
    prob_class_2 = results_df['Prob_Class_2'].tolist()
    prob_class_3 = results_df['Prob_Class_3'].tolist()

    # Form absolute paths for original images
    relative_image_paths = results_df['Filename'].tolist()
    image_paths = []
    for rel_path in relative_image_paths:
        # Ensure path starts with orig_images_base_dir
        if not os.path.isabs(rel_path):
            image_paths.append(os.path.join(orig_images_base_dir, rel_path))
        else:
            image_paths.append(rel_path)

    # Detect the type of CAM method used
    cam_method_name = "CAM"  # Default title
    gradcam_paths = []
    
    if with_heatmap:
        # Find all possible CAM heatmap files in the subfolder
        all_files_in_subfolder = os.listdir(subfolder_path)
        
        # For each original image, find the corresponding heatmap (looking for files containing base_name and "cam")
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            found_heatmap = False
            matched_file = None
            
            # Look for any file containing base_name and "cam" (case insensitive)
            for file in all_files_in_subfolder:
                if base_name in file and "cam" in file.lower():
                    gradcam_paths.append(os.path.join(subfolder_path, file))
                    found_heatmap = True
                    matched_file = file
                    break
            
            # If a matching heatmap is found, try to infer the CAM method from the filename
            if found_heatmap and matched_file:
                # Detect CAM method type from the first matched filename
                if len(gradcam_paths) == 1:
                    matched_file_lower = matched_file.lower()
                    if "gradcam+" in matched_file_lower or "gradcamplus" in matched_file_lower:
                        cam_method_name = "GradCAM++"
                    elif "gradcam" in matched_file_lower:
                        cam_method_name = "GradCAM"
                    elif "scorecam" in matched_file_lower:
                        cam_method_name = "ScoreCAM"
                    elif "eigencam" in matched_file_lower:
                        cam_method_name = "EigenCAM"
                    elif "originalcam" in matched_file_lower or "original_cam" in matched_file_lower:
                        cam_method_name = "Original CAM"
                    elif "cam" in matched_file_lower:
                        cam_method_name = "CAM"
            
            # If no matching heatmap is found, use a blank image as a placeholder
            if not found_heatmap:
                print(f"Warning: Heatmap for {base_name} not found, will use a blank image.")
                gradcam_paths.append(None)  # Use None to indicate no heatmap

    num_frames = len(image_paths)
    if num_frames == 0:
        print(f"No valid image data found in {subfolder_path}, skipping.")
        return

    # Create different numbers of subplots depending on whether heatmap is needed
    if with_heatmap:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_orig, ax_heatmap, ax_prob = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_orig, ax_prob = axes

    # Subplot 1: Display the original image
    orig_display = ax_orig.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
    ax_orig.axis('off')
    ax_orig.set_title('Original Image')

    # Subplot 2 (only if with_heatmap=True): Display the CAM heatmap with the detected CAM method name
    if with_heatmap:
        heatmap_display = ax_heatmap.imshow(np.zeros((600, 600, 3), dtype=np.uint8))
        ax_heatmap.axis('off')
        ax_heatmap.set_title(f'{cam_method_name} Heatmap')

    # Last subplot: Display probability curves
    line0, = ax_prob.plot([], [], 'b-o', label='Equiaxed')
    line1, = ax_prob.plot([], [], 'r-o', label='Columnar')
    line2, = ax_prob.plot([], [], 'g-o', label='Background')
    line3, = ax_prob.plot([], [], 'm-o', label='IMC')
    ax_prob.set_xlim(0, num_frames - 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel('Frame')
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title('Microstructure in Images')
    ax_prob.legend()

    # Data containers for updated values
    x_data = []
    prob0_data = []
    prob1_data = []
    prob2_data = []
    prob3_data = []

    def update(frame_idx):
        # Read and update the original image
        orig_img_path = image_paths[frame_idx]
        try:
            orig_img = Image.open(orig_img_path).convert('RGB').resize((600, 600))
        except Exception as e:
            print(f"Error reading original image: {e}")
            orig_img = Image.new('RGB', (600, 600), color='gray')  # Create a gray image as substitute

        orig_display.set_data(np.array(orig_img))

        # If heatmap is needed, read and update the heatmap image
        if with_heatmap:
            gradcam_img_path = gradcam_paths[frame_idx]
            if gradcam_img_path is not None and os.path.exists(gradcam_img_path):
                try:
                    heatmap_img = Image.open(gradcam_img_path).resize((600, 600))
                except Exception as e:
                    print(f"Error reading heatmap: {e}")
                    heatmap_img = Image.new('RGB', (600, 600), color='black')  # Create a black image as substitute
            else:
                # If no heatmap exists, create a blank image
                heatmap_img = Image.new('RGB', (600, 600), color='black')
            
            heatmap_display.set_data(np.array(heatmap_img))

        # Update probability curve data
        x_data.append(frame_idx)
        prob0_data.append(prob_class_0[frame_idx])
        prob1_data.append(prob_class_1[frame_idx])
        prob2_data.append(prob_class_2[frame_idx])
        prob3_data.append(prob_class_3[frame_idx])

        line0.set_data(x_data, prob0_data)
        line1.set_data(x_data, prob1_data)
        line2.set_data(x_data, prob2_data)
        line3.set_data(x_data, prob3_data)

        if with_heatmap:
            return orig_display, heatmap_display, line0, line1, line2, line3
        else:
            return orig_display, line0, line1, line2, line3

    # Generate animation using FuncAnimation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)

    # Save the animation as an MP4 file
    try:
        writer = FFMpegWriter(fps=30)
        anim.save(save_animation_path, writer=writer)
        print(f"Animation saved to: {save_animation_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.close()
    print(f"Finished processing folder: {subfolder_path}\n")


def generate_animations(root_result_dir, orig_images_base_dir, with_heatmap=False):
    """
    Traverse all subfolders under root_result_dir that contain inference_results.csv,
    generate an animation for each subfolder, and save it as animation.mp4.
    If with_heatmap is True, the animation will include the CAM heatmap; otherwise, it will not.
    """
    valid_subfolders = [
        os.path.join(root_result_dir, folder) 
        for folder in os.listdir(root_result_dir)
        if os.path.isdir(os.path.join(root_result_dir, folder)) and 
           os.path.exists(os.path.join(root_result_dir, folder, "inference_results.csv"))
    ]

    for subfolder_path in tqdm(valid_subfolders, desc="Processing folders"):
        save_path = os.path.join(subfolder_path, "animation.mp4")
        create_animation_for_subfolder(subfolder_path, save_path, orig_images_base_dir, with_heatmap=with_heatmap)


def calculate_accuracy_for_folders(parent_folder):
    """
    Calculate the accuracy within each subfolder and aggregate the results into an accuracy_summary.csv file.
    This is applicable when each subfolder contains inference_results.csv and other CSV files.
    Note: This function requires True_Label column in the CSV files.
    """
    results = []
    
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Find all CSV files in the current subfolder
            csv_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.csv')]

            for csv_file in csv_files:
                csv_path = os.path.join(subfolder_path, csv_file)
                try:
                    data = pd.read_csv(csv_path)
                    if 'True_Label' in data.columns and 'Predicted_Label' in data.columns:
                        accuracy = (data['True_Label'] == data['Predicted_Label']).mean()
                        results.append({
                            'Subfolder': subfolder,
                            'CSV_File': csv_file,
                            'Accuracy': accuracy
                        })
                    else:
                        print(f"Warning: {csv_path} missing True_Label column, skipping accuracy calculation.")
                except Exception as e:
                    print(f"Cannot process file {csv_path}: {e}")
    
    if len(results) == 0:
        print("No usable CSV with True_Label found in subfolders for accuracy statistics.")
        return

    result_df = pd.DataFrame(results)
    output_path = os.path.join(parent_folder, "accuracy_summary.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"Subfolder accuracy statistics completed, results saved to: {output_path}")


def compute_overall_balanced_accuracy(result_base_dir):
    """
    Traverse all inference_results.csv files under the specified root directory (including subdirectories),
    extract the true and predicted labels for all classes,
    and compute and output the following metrics:
       - Macro balanced accuracy (balanced_accuracy_score).
       - A confusion matrix, plotted using seaborn and saved.
       - The accuracy for each class (diagonal element divided by the row sum).
    Note: This function requires True_Label column in the CSV files.
    """
    # Get all files named inference_results.csv (including subdirectories)
    csv_pattern = os.path.join(result_base_dir, '**', 'inference_results.csv')
    csv_files = glob.glob(csv_pattern, recursive=True)

    all_true_labels = []
    all_pred_labels = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
                all_true_labels.extend(df['True_Label'].tolist())
                all_pred_labels.extend(df['Predicted_Label'].tolist())
            else:
                print(f"{csv_file} is missing True_Label or Predicted_Label columns, skipping.")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    if not all_true_labels or not all_pred_labels:
        print("No valid label data collected, unable to compute overall metrics.")
        print("Note: This function requires True_Label column in the CSV files.")
        return

    # 1) Find all possible class labels
    unique_labels = sorted(list(set(all_true_labels + all_pred_labels)))
    print(f"Detected class labels: {unique_labels}")
    
    # 2) Compute balanced accuracy
    try:
        macro_accuracy = balanced_accuracy_score(all_true_labels, all_pred_labels)
        print(f"Overall balanced accuracy: {macro_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Error computing balanced accuracy: {e}")
        return

    # 3) Compute the confusion matrix
    try:
        conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)
        print("Confusion matrix:")
        print(conf_matrix)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")
        return

    # 4) Calculate accuracy for each class: diagonal element / row sum
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    per_class_accuracy = np.zeros(len(unique_labels))
    for i in range(len(unique_labels)):
        if row_sums[i] > 0:
            per_class_accuracy[i] = conf_matrix[i, i] / row_sums[i, 0]
    
    # Create class names
    label_names = []
    for i, label in enumerate(unique_labels):
        if label == 0:
            label_names.append("Equiaxed (0)")
        elif label == 1:
            label_names.append("Columnar (1)")
        elif label == 2:
            label_names.append("Background (2)")
        elif label == 3:
            label_names.append("IMC (3)")
        else:
            label_names.append(f"Class {label}")
    
    for i, acc in enumerate(per_class_accuracy):
        print(f"Accuracy for {label_names[i]}: {acc * 100:.2f}%")

    # 5) Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    conf_matrix_path = os.path.join(result_base_dir, "overall_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_path}")