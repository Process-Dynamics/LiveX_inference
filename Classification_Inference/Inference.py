# =============================================================================
# IMPORTS
# =============================================================================
import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import Model
import utils_analysis

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None, loader=None):
        self.image_paths = image_paths
        self.transform = transform
        self.loader = loader if loader is not None else self.default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, image_path

    @staticmethod
    def default_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

# =============================================================================
# SEED SETTING FUNCTION
# =============================================================================
def _set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# SINGLE IMAGE INFERENCE
# =============================================================================
def infer_single_image(image_path, model, transform, class_names):
    """
    Perform inference on a single image
    
    Args:
        image_path: Path to the image file
        model: Loaded PyTorch model
        transform: Image transformation pipeline
        class_names: List of class names for output
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Prepare results
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_name': class_names[predicted_class],
            'confidence': confidence
        }
        
        # Add individual class probabilities
        for i, class_name in enumerate(class_names):
            result[f'prob_{class_name}'] = probabilities[0, i].item()
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# =============================================================================
# PROCESS FOLDER FUNCTION
# =============================================================================
def process_experiment_folder(folder_path, result_folder_path, model, data_transforms, class_names, batch_size):
    """
    Process a folder containing images and save results
    
    Args:
        folder_path: Path to folder containing images
        result_folder_path: Output folder for results
        model: Loaded PyTorch model
        data_transforms: Image transformation pipeline
        class_names: List of class names
        batch_size: Batch size for processing
    """
    folder_name = os.path.basename(folder_path)
    print(f"Processing experiment: {folder_name}")
    
    # Collect all images in the folder
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in {folder_path}, skipping...")
        return
    
    print(f"Found {len(image_paths)} images in {folder_name}")
    
    # Create dataset and dataloader
    inference_dataset = InferenceDataset(image_paths, transform=data_transforms)
    data_loader = DataLoader(
        inference_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Initialize result containers
    image_results = []
    image_probabilities_class0 = []
    image_probabilities_class1 = []
    image_probabilities_class2 = []
    
    # Inference loop
    for images, paths_batch in tqdm(data_loader, desc=f'Processing {folder_name}'):
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
        
        # Record results for each image in batch
        for i in range(len(images)):
            prob0 = probabilities[i, 0].item()
            prob1 = probabilities[i, 1].item()
            prob2 = probabilities[i, 2].item()
            
            image_results.append({
                'Filename': paths_batch[i],
                'Predicted_Label': predicted[i].item(),
                'Predicted_Name': class_names[predicted[i].item()],
                'Prob_Class_0': prob0,
                'Prob_Class_1': prob1,
                'Prob_Class_2': prob2,
                'Confidence': probabilities[i, predicted[i]].item()
            })
            
            image_probabilities_class0.append(prob0)
            image_probabilities_class1.append(prob1)
            image_probabilities_class2.append(prob2)
    
    # Save results to CSV
    results_df = pd.DataFrame(image_results)
    csv_path = os.path.join(result_folder_path, "inference_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Generate probability curves
    try:
        x_indices = range(len(image_probabilities_class0))
        plt.figure(figsize=(12, 6))
        
        plt.plot(x_indices, image_probabilities_class0, marker='o', label='Equiaxed', alpha=0.7)
        plt.plot(x_indices, image_probabilities_class1, marker='^', label='Columnar', alpha=0.7)
        plt.plot(x_indices, image_probabilities_class2, marker='s', label='Background', alpha=0.7)
        
        all_probs = (image_probabilities_class0 + 
                    image_probabilities_class1 + 
                    image_probabilities_class2)
        ymin = max(0, min(all_probs) - 0.05) if all_probs else 0
        ymax = min(1, max(all_probs) + 0.05) if all_probs else 1
        plt.ylim(ymin, ymax)
        
        plt.xlabel('Frame Index')
        plt.ylabel('Probability')
        plt.title(f'Microstructure Class Probabilities - {folder_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        prob_plot_path = os.path.join(result_folder_path, "probability_curves.png")
        plt.savefig(prob_plot_path, dpi=300)
        plt.close()
        print(f"Probability curves saved to {prob_plot_path}")
        
    except Exception as e:
        print(f"Error plotting probability curves for {folder_name}: {e}")

# =============================================================================
# BATCH INFERENCE FUNCTION
# =============================================================================
def run_batch_inference(input_path, result_folder, model_weights_path, num_classes, batch_size, seed, model_type):
    """
    Run inference on images or experiment folders
    
    Args:
        input_path: Path to single image, folder with images, or parent folder containing experiment folders
        result_folder: Output folder for results
        model_weights_path: Path to model weights
        num_classes: Number of classes
        batch_size: Batch size for processing
        seed: Random seed
        model_type: Type of model ("resnet" or "swin")
    """
    _set_seed(seed)
    
    # Build and load model
    if model_type.lower() == "resnet":
        model = Model.build_resnet152_for_xray(
            num_classes=num_classes, pretrained=True, freeze_backbone=False
        )
    elif model_type.lower() == "swin":
        model = Model.build_swin_transformer_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {key.replace('module.', ''): state_dict[key] for key in state_dict}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Define transforms
    data_transforms = transforms.Compose([
        transforms.CenterCrop(1056),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    class_names = ['Equiax', 'Columnar', 'Background']
    os.makedirs(result_folder, exist_ok=True)
    
    # Check if input is single image
    if os.path.isfile(input_path) and input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Single image inference
        print(f"Processing single image: {input_path}")
        result = infer_single_image(input_path, model, data_transforms, class_names)
        if result:
            print(f"Prediction: {result['predicted_name']} (confidence: {result['confidence']:.3f})")
            
            # Save result to CSV
            df = pd.DataFrame([result])
            csv_path = os.path.join(result_folder, "single_image_result.csv")
            df.to_csv(csv_path, index=False)
            print(f"Result saved to {csv_path}")
        return
    
    # Check if input is a directory
    if not os.path.isdir(input_path):
        print(f"Input path {input_path} is neither a valid image file nor a directory")
        return
    
    # Check for images directly in the input folder
    direct_images = []
    subdirectories = []
    
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            direct_images.append(item_path)
        elif os.path.isdir(item_path):
            subdirectories.append(item)
    
    if direct_images:
        # Input folder directly contains images - create a subfolder for results
        input_folder_name = os.path.basename(input_path)
        experiment_result_folder = os.path.join(result_folder, input_folder_name)
        os.makedirs(experiment_result_folder, exist_ok=True)
        
        print(f"Input folder contains {len(direct_images)} images directly")
        print(f"Creating results subfolder: {input_folder_name}")
        
        process_experiment_folder(input_path, experiment_result_folder, model, data_transforms, class_names, batch_size)
        
    elif subdirectories:
        # Input folder contains subdirectories - process each as separate experiment
        experiment_folders = []
        for subdir in subdirectories:
            subdir_path = os.path.join(input_path, subdir)
            # Check if subdirectory contains images
            subdir_images = []
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        subdir_images.append(os.path.join(root, file))
            if subdir_images:
                experiment_folders.append(subdir)
        
        if not experiment_folders:
            print(f"No images found in any subdirectories of {input_path}")
            return
        
        print(f"Found {len(experiment_folders)} experiment folders")
        
        # Process each experiment folder
        for experiment_folder in experiment_folders:
            experiment_path = os.path.join(input_path, experiment_folder)
            experiment_result_folder = os.path.join(result_folder, experiment_folder)
            os.makedirs(experiment_result_folder, exist_ok=True)
            
            process_experiment_folder(experiment_path, experiment_result_folder, model, data_transforms, class_names, batch_size)
    
    else:
        print(f"No images found in {input_path} or its subdirectories")
        return

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main(
    INPUT_PATH,
    RESULT_FOLDER, 
    MODEL_WEIGHTS_PATH, 
    num_classes, 
    batch_size, 
    seed, 
    model_type,
    generate_animations=False
):
    """
    Main inference function
    
    Args:
        INPUT_PATH: Path to single image, folder with images, or parent folder with experiment folders
        RESULT_FOLDER: Output folder for results
        MODEL_WEIGHTS_PATH: Path to model weights
        num_classes: Number of classes
        batch_size: Batch size for processing
        seed: Random seed
        model_type: Model type ("resnet" or "swin")
        generate_animations: Whether to generate animations using utils_analysis
    """
    
    # Run inference
    try:
        run_batch_inference(
            INPUT_PATH, RESULT_FOLDER, MODEL_WEIGHTS_PATH,
            num_classes, batch_size, seed, model_type
        )
        print("Inference complete!")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate animations if requested and if we processed multiple experiments
    if generate_animations:
        try:
            print("\nGenerating animations...")
            utils_analysis.generate_animations(RESULT_FOLDER, INPUT_PATH, with_heatmap=False)
            print("Animation generation complete")
        except Exception as e:
            print(f"Error generating animations: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll tasks completed!")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Configuration parameters
    INPUT_PATH = r'/home/shun/Project/Grains-Classification_test/Test_folder/ma2035_005'  # Single image, folder with images, or parent folder
    RESULT_FOLDER = r'/home/shun/Project/Grains-Classification_test/Result'
    MODEL_WEIGHTS_PATH = r'/home/shun/Project/Grains-Classification_test/Model/Swin Transformer.pth' #Change the path when the model_type changes.
    num_classes = 3
    batch_size = 1
    seed = 43
    model_type = "swin"  # Options: "resnet", "swin"
    
    # Execution flags
    generate_animations = True  # Set to True if you want animations
    
    # Run inference
    main(
        INPUT_PATH, RESULT_FOLDER, MODEL_WEIGHTS_PATH,
        num_classes, batch_size, seed, model_type,
        generate_animations
    )