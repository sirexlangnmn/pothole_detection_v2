"""
AyosBayan ML Module - Utility Functions
========================================

This file contains helper functions used across the ML module.
These utilities support training, inference, and API operations.

Author: Federex (AyosBayan Research Project)
Version: 1.0
Last Updated: April 2026
"""

# Import required libraries for the AyosBayan project
# ================================================

# os - Operating system interface for file path operations
# Used for: checking if model file exists, path joining
import os

# typing - Type hints for better code documentation
# Used for: defining return types of functions
from typing import Dict, List, Optional, Tuple

# numpy - Numerical computing library
# Used for: handling numerical data from image processing
import numpy as np

# ultralytics - YOLOv8 library for object detection
# This is the core ML library for pothole detection
from ultralytics import YOLO

# pillow (PIL) - Python Imaging Library
# Used for: loading, saving, and manipulating images
from PIL import Image


# ============================================================================
# CONSTANTS - Configuration values used throughout the module
# ============================================================================

# CONFIDENCE_THRESHOLD - Minimum confidence score to consider a detection valid
# 
# Purpose: This threshold helps filter out low-confidence detections that might
# be false positives. In the context of AyosBayan:
#   - If confidence >= 0.5: Report is auto-verified as real pothole
#   - If confidence < 0.5 or no detection: Flagged as potential fake report
#     that needs manual review by LGU administrators
#
# Why 0.5? This is a standard threshold that balances:
#   - Not missing real potholes (high recall)
#   - Not flagging too many false positives (reasonable precision)
# Can be tuned higher (e.g., 0.7) for stricter verification
CONFIDENCE_THRESHOLD = 0.5

# DEFAULT_MODEL_PATH - Path to the trained YOLOv8 model
# 
# This is the path where the trained model weights are stored.
# The model file 'best.pt' contains the learned weights from training
# on the pothole dataset.
#
# In production, this could also be a path to a remote model (URL)
DEFAULT_MODEL_PATH = "best.pt"

# CLASS_NAMES - List of object classes the model can detect
# 
# For this version 1.0, we only detect potholes.
# Future versions (v2.0) may detect multiple classes like:
#   - crack: Road cracks
#   - bump: Speed bumps
#   - manhole: Open manholes
#   - flood: Water flooding
CLASS_NAMES = ["pothole"]


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Load a trained YOLOv8 model for inference.
    
    This function loads the trained YOLOv8 model from a .pt file.
    The model contains the neural network weights learned during training
    on the pothole dataset.
    
    How it works:
    1. Check if the model file exists at the given path
    2. Load the model using Ultralytics YOLO class
    3. Return the loaded model for inference
    
    Args:
        model_path: Path to the trained model file (.pt format)
                   If not provided, uses DEFAULT_MODEL_PATH
    
    Returns:
        YOLO: A loaded YOLO model ready for inference
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If there's an issue loading the model
    
    Example:
        >>> # Load the default pothole detection model
        >>> model = load_model()
        >>> 
        >>> # Load a specific version
        >>> model = load_model("best_v2.pt")
    """
    # First, check if the model file exists
    # This prevents confusing errors if the file is missing
    if not os.path.exists(model_path):
        # Raise a clear error message explaining what went wrong
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Please ensure you have trained the model first.\n"
            f"Run: python src/train.py\n"
            f"Or download a pre-trained model."
        )
    
    # Load the YOLOv8 model using Ultralytics
    # The YOLO class automatically:
    #   - Parses the model architecture from the .pt file
    #   - Loads the pre-trained weights
    #   - Prepares the model for inference mode
    model = YOLO(model_path)
    
    # Return the loaded model - ready to use for predictions
    return model


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path.
    
    This function loads an image file (JPEG, PNG, etc.) and converts it
    to a PIL Image object that can be used for inference.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image.Image: A PIL Image object
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the image cannot be loaded (corrupted format, etc.)
    
    Example:
        >>> # Load a citizen-reported image
        >>> image = load_image("tests/test_images/pothole_example.jpg")
        >>> print(f"Image size: {image.size}")
    """
    # Check if the image file exists before trying to load
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Use PIL (Pillow) to open and load the image
    # Image.open() automatically detects the format (JPEG, PNG, etc.)
    # and creates the appropriate image object
    image = Image.open(image_path)
    
    # Return the loaded PIL Image object
    # This can be passed directly to YOLO for inference
    return image


def validate_image(image_path: str) -> Tuple[bool, str]:
    """
    Validate an image file before inference.
    
    This function checks if an image file is valid and suitable for
    pothole detection. It performs several checks:
    1. File existence
    2. File extension (supported formats)
    3. File is not empty
    4. Can be opened as an image
    
    Why validate? This prevents errors during inference and provides
    helpful feedback to the citizen if their upload is invalid.
    
    Args:
        image_path: Path to the image file to validate
    
    Returns:
        Tuple[bool, str]: A tuple containing:
            - is_valid: True if image is valid, False otherwise
            - message: Explanation of the result (for user feedback)
    
    Example:
        >>> # Validate a citizen's upload
        >>> is_valid, message = validate_image("upload.jpg")
        >>> if is_valid:
        >>>     print("Image is valid! Proceeding with detection...")
        >>> else:
        >>>     print(f"Invalid image: {message}")
    """
    # Check 1: Does the file exist?
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}"
    
    # Check 2: Is the file extension supported?
    # YOLOv8 supports: JPEG, PNG, BMP, WEBP, TIFF
    # We check the lowercase extension to handle case variations
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    if not image_path.lower().endswith(supported_extensions):
        return False, (
            f"Unsupported image format. "
            f"Supported formats: JPEG, PNG, BMP, WEBP, TIFF"
        )
    
    # Check 3: Is the file empty?
    # If the file size is 0 bytes, it's definitely corrupted or invalid
    if os.path.getsize(image_path) == 0:
        return False, "Image file is empty (0 bytes)"
    
    # Check 4: Can we open it as an image?
    # This catches corrupted files that have valid extensions but
    # contain invalid/corrupted image data
    try:
        # Try to open the image using PIL
        with Image.open(image_path) as img:
            # Load the image data to verify it's valid
            img.verify()
    except Exception as e:
        # If any error occurs, the image is invalid
        return False, f"Cannot open image: {str(e)}"
    
    # All checks passed - the image is valid
    return True, "Image is valid"


# ============================================================================
# INFERENCE RESULT PROCESSING FUNCTIONS
# ============================================================================

def process_detection_result(results: list) -> Dict:
    """
    Process YOLOv8 inference results into a standardized format.
    
    This function takes the raw output from YOLOv8 inference and
    converts it into a clean, easy-to-use dictionary format.
    
    The raw result contains:
    - Bounding box coordinates
    - Confidence scores
    - Class predictions
    - And more metadata
    
    We simplify this to just what the AyosBayan API needs:
    - detected: boolean (was a pothole found?)
    - confidence: float (how confident is the model?)
    - bounding_box: optional coordinates for visualization
    
    Args:
        results: List of results from YOLO model inference
                 Typically contains one result per image
    
    Returns:
        Dict: A dictionary containing:
            - detected (bool): Whether a pothole was detected
            - confidence (float): Confidence score (0.0 - 1.0)
            - bounding_box (dict or None): Box coordinates if detected
            - is_potential_fake (bool): Whether report needs manual review
    
    Example:
        >>> # Run inference
        >>> results = model.predict("test.jpg")
        >>> 
        >>> # Process results
        >>> output = process_detection_result(results)
        >>> print(f"Detected: {output['detected']}")
        >>> print(f"Confidence: {output['confidence']}")
    """
    # Initialize default values (no detection)
    # These will be updated if a pothole is found
    detected = False
    confidence = 0.0
    bounding_box = None
    
    # Check if YOLOv8 found any objects in the image
    # results[0] contains the first image's detection results
    # .boxes contains all detected bounding boxes
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # YOLOv8 found at least one object!
        # (This is usually a pothole, since that's what we trained on)
        
        # Get the first detection (most confident)
        # In practice, we take the highest confidence detection
        box = results[0].boxes[0]
        
        # Mark as detected
        detected = True
        
        # Get the confidence score (0.0 to 1.0)
        # This tells us how confident the model is about this detection
        # 1.0 = 100% confident, 0.0 = 0% confident
        confidence = float(box.conf[0])
        
        # Extract bounding box coordinates
        # YOLOv8 provides boxes in.xyxy format (not xywh)
        # xyxy = [x1, y1, x2, y2] where:
        #   - (x1, y1) = top-left corner
        #   - (x2, y2) = bottom-right corner
        # These are in pixel coordinates relative to the image
        xyxy = box.xyxy[0].cpu().numpy()
        
        # Convert to a dictionary for cleaner API responses
        bounding_box = {
            # x1 = left coordinate (pixels from left edge)
            "x1": int(xyxy[0]),
            # y1 = top coordinate (pixels from top edge)
            "y1": int(xyxy[1]),
            # x2 = right coordinate
            "x2": int(xyxy[2]),
            # y2 = bottom coordinate
            "y2": int(xyxy[3]),
            # Calculate width and height for convenience
            "width": int(xyxy[2] - xyxy[0]),
            "height": int(xyxy[3] - xyxy[1])
        }
    
    # Determine if this is a potential fake report
    # 
    # A report is flagged as "potential fake" if:
    # 1. No pothole was detected (detected = False), OR
    # 2. Confidence is below the threshold (confidence < 0.5)
    #
    # These reports need manual review by LGU administrators
    # to determine if the report is legitimate or fake.
    is_potential_fake = not detected or confidence < CONFIDENCE_THRESHOLD
    
    # Return all the processed information as a dictionary
    return {
        "detected": detected,                    # Was a pothole found?
        "confidence": confidence,                # How confident (0.0-1.0)
        "bounding_box": bounding_box,             # Where in the image?
        "is_potential_fake": is_potential_fake   # Needs manual review?
    }


def create_api_response(detection_result: Dict) -> Dict:
    """
    Create the final API response for the /detect-pothole endpoint.
    
    This function takes the processed detection result and formats it
    into the final JSON response that will be sent to the client (LGU dashboard).
    
    The API response includes:
    - detected: Whether a pothole was found
    - confidence: Confidence score (0.0 - 1.0)
    - is_potential_fake: Whether the report needs manual review
    - auto_tag: The tag/classification for the report
    - message: Human-readable status message
    
    Args:
        detection_result: Dictionary from process_detection_result()
    
    Returns:
        Dict: Final API response ready to be JSON-serialized
    
    Example:
        >>> # After running inference
        >>> detection_result = process_detection_result(results)
        >>> 
        >>> # Create API response
        >>> api_response = create_api_response(detection_result)
        >>> print(api_response)
        >>> # {
        >>> #     "detected": True,
        >>> #     "confidence": 0.94,
        >>> #     "is_potential_fake": False,
        >>> #     "auto_tag": "pothole",
        >>> #     "message": "Pothole detected with high confidence"
        >>> # }
    """
    # Extract values from the detection result
    detected = detection_result["detected"]
    confidence = detection_result["confidence"]
    is_potential_fake = detection_result["is_potential_fake"]
    
    # Determine the auto_tag based on detection
    # If pothole detected: auto_tag = "pothole"
    # If no pothole: auto_tag = "normal" (no damage detected)
    auto_tag = "pothole" if detected else "normal"
    
    # Create a human-readable message based on the result
    # This provides clear feedback for the LGU dashboard
    if detected and confidence >= CONFIDENCE_THRESHOLD:
        # High confidence detection - report auto-verified
        message = f"Pothole detected with high confidence ({confidence:.2%})"
    elif detected and confidence < CONFIDENCE_THRESHOLD:
        # Low confidence detection - needs review
        message = f"Pothole detected but confidence is low ({confidence:.2%})"
    else:
        # No detection - might be fake report
        message = "No pothole detected - report requires manual review"
    
    # Build the final API response dictionary
    # All field names are snake_case for Python convention
    api_response = {
        "detected": detected,                                    # True/False
        "confidence": confidence,                                # 0.0 - 1.0
        "is_potential_fake": is_potential_fake,                  # Needs review?
        "auto_tag": auto_tag,                                     # "pothole" or "normal"
        "message": message,                                      # Human-readable
        "bounding_box": detection_result.get("bounding_box")     # Optional visualization
    }
    
    return api_response


# ============================================================================
# UTILITY FUNCTIONS FOR FILE OPERATIONS
# ============================================================================

def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    This is a helper function that creates directories if they don't exist.
    Useful for:
    - Creating output directories for inference results
    - Creating directories for saved models
    - Ensuring the data directory structure exists
    
    Args:
        directory_path: Path to the directory to ensure exists
    
    Example:
        >>> # Ensure output directory exists
        >>> ensure_directory("output/pothole_detections")
        >>> # Now we can safely save files to this directory
    """
    # os.makedirs creates all intermediate directories in the path
    # exist_ok=True means it won't raise an error if directory exists
    os.makedirs(directory_path, exist_ok=True)


def get_model_info(model_path: str = DEFAULT_MODEL_PATH) -> Dict:
    """
    Get information about a trained model.
    
    This function loads a model and returns metadata about it,
    such as:
    - Model type (YOLOv8n, YOLOv8s, etc.)
    - Number of parameters
    - Input image size
    - Classes the model can detect
    
    This is useful for debugging and API documentation.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Dict: Information about the model
    
    Example:
        >>> info = get_model_info()
        >>> print(f"Model type: {info['model_type']}")
        >>> print(f"Classes: {info['class_names']}")
    """
    # Load the model (will raise error if file doesn't exist)
    model = load_model(model_path)
    
    # Get model information from the model object
    # YOLO models have a .model attribute with detailed info
    model_info = {
        # Model type (e.g., "yolov8n", "yolov8s")
        "model_type": model.model_name if hasattr(model, 'model_name') else "YOLOv8",
        # Number of classes (1 for pothole-only detection)
        "num_classes": len(CLASS_NAMES),
        # List of class names
        "class_names": CLASS_NAMES,
        # Confidence threshold used
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        # Default input image size (YOLOv8 standard)
        "input_image_size": 640
    }
    
    return model_info


# ============================================================================
# MAIN BLOCK - For testing utilities
# ============================================================================

if __name__ == "__main__":
    """
    Main block for testing utility functions.
    
    Run this file directly to test the utility functions:
        python src/utils.py
    
    This will:
    1. Print the current configuration (threshold, model path, etc.)
    2. Test the validate_image function with a sample image
    3. Display model information
    """
    
    # Print a header for the test output
    print("=" * 60)
    print("AyosBayan ML Module - Utility Functions Test")
    print("=" * 60)
    
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"  - Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  - Default Model Path: {DEFAULT_MODEL_PATH}")
    print(f"  - Class Names: {CLASS_NAMES}")
    
    # Test image validation with a sample image
    # Note: This test will fail if the test image doesn't exist
    # which is expected before training/inference
    print("\nTesting Image Validation:")
    test_image_path = "tests/test_images/pothole_example.jpg"
    is_valid, message = validate_image(test_image_path)
    print(f"  - Test image: {test_image_path}")
    print(f"  - Valid: {is_valid}")
    print(f"  - Message: {message}")
    
    # Test model info loading
    # Note: This will fail if the model doesn't exist yet
    print("\nTesting Model Info Loading:")
    try:
        info = get_model_info()
        print(f"  - Model loaded successfully!")
        print(f"  - Model Type: {info['model_type']}")
        print(f"  - Number of Classes: {info['num_classes']}")
        print(f"  - Classes: {info['class_names']}")
    except FileNotFoundError as e:
        print(f"  - Error: {e}")
        print("  - This is expected if no model has been trained yet.")
        print("  - Run: python src/train.py to train the model")
    
    # Print completion message
    print("\n" + "=" * 60)
    print("Utility functions test complete!")
    print("=" * 60)
