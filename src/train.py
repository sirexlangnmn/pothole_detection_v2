"""
AyosBayan ML Module - Training Script
=====================================

This script trains the YOLOv8 model for pothole detection.
It handles the complete training pipeline including:
- Loading the pre-trained YOLOv8 model
- Configuring training parameters
- Running the training process
- Saving the best model

The training uses transfer learning from COCO-pretrained weights,
which already knows how to detect general objects (roads, vehicles, etc.)
We fine-tune it specifically for pothole detection.

Author: Federex (AyosBayan Research Project)
Version: 1.0
Last Updated: April 2026

Usage:
    python src/train.py
    
Or with custom parameters:
    python src/train.py --data data/pothole_dataset/data.yaml --epochs 100
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

# Import standard library modules
import os              # Operating system interfaces (file paths, environment)
import sys             # System-specific parameters and functions
import argparse       # Command-line argument parsing

# Import YOLOv8 from Ultralytics
# This is the core ML library for object detection
# Ultralytics provides pre-trained models and training utilities
from ultralytics import YOLO


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# These are the default values used when no command-line arguments are provided
# They follow the recommendations from the README

# DATA_CONFIG - Path to the dataset configuration file
# This YAML file contains information about:
#   - Path to training images
#   - Path to validation images
#   - Number of classes
#   - Class names (pothole)
DATA_CONFIG = "data/pothole_dataset/data.yaml"

# MODEL_VARIANT - Which YOLOv8 variant to use
# Options: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
# 
# yolov8n (nano) is chosen because:
#   - Fastest inference (< 2 seconds on CPU)
#   - Smallest model size (easy to deploy)
#   - Good accuracy for v1.0 (can upgrade to 's' if needed)
# 
# The model hierarchy (larger = more accurate but slower):
#   yolov8n < yolov8s < yolov8m < yolov8l < yolov8x
MODEL_VARIANT = "yolov8n"

# EPOCHS - Number of training epochs
# An epoch = one complete pass through the entire training dataset
# 
# Why 50?
#   - YOLOv8 converges relatively quickly due to transfer learning
#   - Pre-trained weights from COCO already "know" basic features
#   - 50 epochs is a good balance between training time and accuracy
#   - Can increase to 100 if more training is needed
#   - Can decrease to 20 for quick testing
EPOCHS = 50

# IMAGE_SIZE - Input image size for training
# YOLOv8 works best with 640x640 pixel images
# This is the standard YOLOv8 input size
# 
# Why 640?
#   - Optimized for YOLOv8 architecture
#   - Good balance between detail and processing speed
#   - Smaller sizes (320) are faster but less accurate
#   - Larger sizes (1024) are more accurate but slower
IMAGE_SIZE = 640

# BATCH_SIZE - Number of images processed per training step
# 
# Why 16?
#   - Good balance between memory usage and training stability
#   - Works on most GPUs (8GB+ VRAM)
#   - If you get out-of-memory errors, reduce to 8 or 4
#   - If using Google Colab free tier, try 8 or 16
BATCH_SIZE = 16

# MODEL_NAME - Name for the output model file
# This will be saved as best.pt in the project root
MODEL_NAME = "ayosbayan_pothole"

# PROJECT_DIR - Directory to save training results
# The results will be saved in: runs/train/ayosbayan_pothole/
PROJECT_DIR = "runs"


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments for custom training configuration.
    
    This allows users to customize training without editing the code.
    They can specify custom data, epochs, model, etc.
    
    Returns:
        argparse.Namespace: Parsed arguments with their values
    
    Example:
        python src/train.py --epochs 100 --batch-size 8
    """
    # Create the argument parser
    # description explains what the script does
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 pothole detection model for AyosBayan"
    )
    
    # Add --data argument (path to data.yaml)
    # This is required to tell the model where the dataset is
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_CONFIG,
        help="Path to the dataset configuration YAML file"
    )
    
    # Add --model argument (model variant)
    # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_VARIANT,
        help="YOLOv8 model variant (n/s/m/l/x)"
    )
    
    # Add --epochs argument (number of training epochs)
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs (default: 50)"
    )
    
    # Add --imgsz argument (image size)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help="Input image size (default: 640)"
    )
    
    # Add --batch-size argument
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (default: 16)"
    )
    
    # Add --name argument (model name for saving)
    parser.add_argument(
        "--name",
        type=str,
        default=MODEL_NAME,
        help="Name for the training run"
    )
    
    # Add --workers argument (data loading workers)
    # More workers = faster data loading but more memory
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    
    # Add --patience argument (early stopping patience)
    # If no improvement for this many epochs, stop training
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    
    # Add --project argument (project directory)
    parser.add_argument(
        "--project",
        type=str,
        default=PROJECT_DIR,
        help="Directory to save training results"
    )
    
    # Add --exist-ok argument (overwrite existing)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing results if name matches"
    )
    
    # Return parsed arguments
    return parser.parse_args()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_yolov8(args):
    """
    Train YOLOv8 model for pothole detection.
    
    This is the main training function that:
    1. Loads the pre-trained YOLOv8 model
    2. Configures training parameters
    3. Runs the training process
    4. Saves the best model
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        None: Training results are saved to disk
    
    How it works:
    ----------------
    1. First, we load a pre-trained YOLOv8 model (transfer learning)
       - This starts with a model that already knows basic visual features
       - Pre-trained on COCO dataset (contains 80+ common objects)
       - We only need to fine-tune for potholes
    
    2. We configure training with the dataset
       - data.yaml tells YOLO where images and labels are
       - Training uses 70% of data, validation uses 20%
       - Testing uses remaining 10%
    
    3. Training process:
       - For each epoch, the model sees all training images
       - It learns to predict bounding boxes around potholes
       - After each epoch, it's evaluated on validation data
       - The "best" model is saved (highest validation accuracy)
    
    4. After training:
       - best.pt is saved (the trained model weights)
       - Training logs are saved in runs/ directory
       - We can now use this model for inference
    """
    # Print a banner to show training is starting
    print("=" * 60)
    print("AyosBayan ML - Pothole Detection Model Training")
    print("=" * 60)
    
    # Print the training configuration
    # This helps users understand what's being used
    print("\nTraining Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Name: {args.name}")
    
    # Check if the dataset configuration file exists
    # The data.yaml file is required for training
    if not os.path.exists(args.data):
        # If the file doesn't exist, show a helpful error message
        print(f"\nERROR: Dataset configuration not found!")
        print(f"Please ensure '{args.data}' exists.")
        print(f"\nTo create the dataset configuration:")
        print(f"  1. Download or prepare your pothole dataset")
        print(f"  2. Place images in data/pothole_dataset/images/")
        print(f"  3. Place labels in data/pothole_dataset/labels/")
        print(f"  4. Create data/pothole_dataset/data.yaml")
        sys.exit(1)
    
    # Step 1: Load the pre-trained YOLOv8 model
    # 
    # We use transfer learning by starting with a pre-trained model.
    # This model was trained on COCO dataset and already knows
    # how to detect many objects (cars, roads, etc.).
    #
    # The model file will be automatically downloaded if not present.
    # Downloaded models are cached in ~/.ultralytics/
    print(f"\nLoading pre-trained {args.model}.pt model...")
    
    # Create a YOLO model instance
    # This loads the model architecture and pre-trained weights
    model = YOLO(f"{args.model}.pt")
    
    # Step 2: Configure and run training
    # 
    # The .train() method is the main training entry point.
    # It handles:
    #   - Data loading and preprocessing
    #   - Forward pass (computing predictions)
    #   - Loss calculation
    #   - Backward pass (updating weights)
    #   - Validation after each epoch
    #   - Checkpoint saving
    #
    # All parameters are passed as keyword arguments.
    print("\nStarting training...")
    print("-" * 40)
    
    # Run the training
    # This is where the actual learning happens
    # Training can take 10 minutes to several hours
    # depending on epochs, GPU, and dataset size
    results = model.train(
        # Data configuration - path to data.yaml
        # This tells YOLO where to find images and labels
        data=args.data,
        
        # Number of training epochs
        # More epochs = more training but longer time
        # We start with the pre-trained weights, so 50 is usually enough
        epochs=args.epochs,
        
        # Input image size
        # YOLOv8 expects square images of this size
        # 640 is the standard size that balances speed and accuracy
        imgsz=args.imgsz,
        
        # Batch size - how many images per training step
        # Larger batches = more stable training but more memory
        # 16 is a good default for most GPUs
        batch=args.batch_size,
        
        # Project directory - where to save results
        # All outputs go into runs/train/{name}/
        project=args.project,
        
        # Name of this training run
        # Results saved to runs/train/{name}/
        name=args.name,
        
        # Overwrite if results already exist
        # Set to True to allow re-training with same name
        exist_ok=args.exist_ok,
        
        # Number of workers for data loading
        # More workers = faster data loading
        # But too many can cause memory issues
        workers=args.workers,
        
        # Early stopping patience
        # If validation loss doesn't improve for this many epochs,
        # training stops early to prevent overfitting
        patience=args.patience,
        
        # Print training progress every N iterations
        # Set to 10 to see progress more frequently
        verbose=True,
        
        # Device to use for training
        # 'cuda' = NVIDIA GPU (fastest)
        # 'cpu' = CPU only (slow)
        # Let YOLO auto-detect the best option
        device='',  # Auto-detect (cuda or cpu)
        
        # Training cache - saves images in memory for faster training
        # Set to False to use less memory but slower training
        cache=False,
        
        # Image augmentation settings
        # YOLOv8 has built-in augmentation:
        # - Mosaic: combines 4 images into 1 (helps learning)
        # - Copy-paste: copies objects between images
        # - MixUp: blends two images
        # - HSV: adjusts hue, saturation, value (color changes)
        # - Flip: flips images horizontally
        # - Rotation: rotates images slightly
        # 
        # These help the model generalize to different conditions
        # (rain, different lighting, angles, etc.)
        amp=True,  # Automatic Mixed Precision (faster training)
    )
    
    # Print completion message
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Tell the user where the trained model is saved
    # The best model weights are saved in the runs directory
    model_path = f"{args.project}/train/{args.name}/weights/best.pt"
    print(f"\nTrained model saved to: {model_path}")
    
    # Explain next steps
    print("\nNext Steps:")
    print(f"  1. Copy best.pt to project root: cp {model_path} ./best.pt")
    print(f"  2. Test the model: python src/inference.py --image tests/test_images/<image>.jpg")
    print(f"  3. Start the API: uvicorn src.api:app --reload --port 8000")
    
    # Return the results for further analysis if needed
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the training script.
    
    When this script is run directly (not imported), it:
    1. Parses command-line arguments
    2. Calls the training function
    3. Handles any errors gracefully
    
    Usage:
        python src/train.py
        
    Or with custom parameters:
        python src/train.py --epochs 100 --batch-size 8 --data custom/path/data.yaml
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Run the training
        train_yolov8(args)
        
    except KeyboardInterrupt:
        # Handle user pressing Ctrl+C
        # This allows graceful interruption of training
        print("\n\nTraining interrupted by user.")
        print("Partial results may have been saved.")
        sys.exit(0)
        
    except Exception as e:
        # Handle any other errors
        # Print a helpful error message
        print(f"\nERROR: Training failed with exception:")
        print(f"  {str(e)}")
        print(f"\nPlease check:")
        print(f"  1. Dataset path is correct")
        print(f"  2. Sufficient disk space for training outputs")
        print(f"  3. Enough RAM/VRAM for batch size")
        sys.exit(1)
