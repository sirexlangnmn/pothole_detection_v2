"""
AyosBayan ML Module - Local Inference Script
============================================

This script performs local inference (testing) on images using the trained
YOLOv8 pothole detection model. It's designed for local testing of the ML
module before deploying to production.

This script is typically used by:
- Developers testing the model locally
- Quality assurance testing with test images
- Batch processing of multiple images
- Debugging and troubleshooting

Author: Federex (AyosBayan Research Project)
Version: 1.0
Last Updated: April 2026

Usage:
    python src/inference.py --image tests/test_images/pothole_example.jpg
    
Or with custom model:
    python src/inference.py --image test.jpg --model best_v2.pt
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

# Import standard library modules
import os             # Operating system interfaces (file paths, etc.)
import sys            # System-specific parameters and functions
import argparse      # Command-line argument parsing
from pathlib import Path  # Object-oriented file paths

# Import our utility functions
# These are defined in src/utils.py and help with:
# - Loading the model
# - Validating images
# - Processing detection results
from utils import (
    load_model,
    validate_image,
    process_detection_result,
    create_api_response,
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH
)

# Import YOLO from Ultralytics
# This is the core ML library for object detection
from ultralytics import YOLO


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

# Default paths and settings
# These can be overridden via command-line arguments

# DEFAULT_IMAGE_PATH - Path to a sample test image
# This is used when no image is specified
DEFAULT_IMAGE_PATH = "tests/test_images/pothole_example.jpg"

# DEFAULT_OUTPUT_DIR - Where to save inference results
# Results include annotated images with bounding boxes
DEFAULT_OUTPUT_DIR = "output"


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments for inference configuration.
    
    This allows users to:
    - Specify which image to test
    - Use a different model file
    - Save the annotated output image
    - Control confidence threshold
    
    Returns:
        argparse.Namespace: Parsed arguments
    
    Example:
        python src/inference.py --image test.jpg --model best_v2.pt --save
    """
    # Create the argument parser with a description
    parser = argparse.ArgumentParser(
        description="Run pothole detection inference on local images"
    )
    
    # Add --image argument (required - path to input image)
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input image file"
    )
    
    # Add --model argument (path to model file)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model file (.pt)"
    )
    
    # Add --save argument (save annotated image)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated image with bounding boxes"
    )
    
    # Add --output argument (output directory)
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output images"
    )
    
    # Add --conf argument (confidence threshold)
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})"
    )
    
    # Add --verbose argument (print detailed output)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed inference information"
    )
    
    # Return parsed arguments
    return parser.parse_args()


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def run_inference(model: YOLO, image_path: str, args) -> dict:
    """
    Run pothole detection inference on a single image.
    
    This function:
    1. Validates the input image
    2. Runs the YOLO model on the image
    3. Processes the results
    4. Returns the detection results
    
    Args:
        model: A loaded YOLO model ready for inference
        image_path: Path to the image file to analyze
        args: Parsed command-line arguments
    
    Returns:
        dict: Detection results including:
            - detected: bool
            - confidence: float
            - is_potential_fake: bool
            - bounding_box: dict or None
    
    How it works:
    --------------
    1. First, we validate the image to make sure it's valid
       - Check file exists
       - Check format is supported
       - Check file is not corrupted
    
    2. Then we run the model.predict() method
       - This performs forward pass through the neural network
       - Returns bounding boxes, confidence scores, class IDs
       - Takes < 2 seconds on CPU
    
    3. Finally, we process the raw results
       - Convert to a clean dictionary format
       - Add the fake report detection logic
    """
    # Step 1: Validate the input image
    # Before running inference, we check if the image is valid
    # This prevents confusing errors and provides clear feedback
    print(f"\nValidating image: {image_path}")
    is_valid, message = validate_image(image_path)
    
    if not is_valid:
        # If image is invalid, raise an error with helpful message
        raise ValueError(f"Invalid image: {message}")
    
    print(f"Image validation passed!")
    
    # Step 2: Run inference with the YOLO model
    # 
    # The model.predict() method is the main inference function.
    # It:
    #   - Loads the image
    #   - Preprocesses it (resizes to 640x640)
    #   - Runs the neural network forward pass
    #   - Returns detection results
    #
    # Key parameters:
    #   - conf: Minimum confidence threshold
    #     (detections below this are ignored)
    #   - iou: Intersection over Union threshold
    #     (helps remove duplicate detections)
    #   - save: Whether to save annotated image
    #   - save_txt: Whether to save results as text
    #   - verbose: Whether to print detailed output
    print(f"Running inference...")
    
    # Run the prediction
    # results is a list (one entry per image)
    results = model.predict(
        source=image_path,         # Input image path
        conf=args.conf,             # Confidence threshold
        iou=0.7,                    # IOU threshold for NMS
        save=args.save,             # Save annotated image?
        save_txt=False,             # Don't save text results
        save_conf=True,             # Save confidence in text
        verbose=args.verbose,      # Print detailed output?
        device='',                  # Auto-detect (cpu or cuda)
        show=False,                 # Don't display window (for server)
        line_thickness=3,           # Bounding box line thickness
        hide_labels=False,          # Show class labels on boxes
        project=args.output,       # Output directory
        name='inference'           # Subdirectory name
    )
    
    # Step 3: Process the results
    # The raw results from YOLO need to be processed into
    # a clean, standardized format for our API
    detection_result = process_detection_result(results)
    
    return detection_result


def print_detection_result(result: dict) -> None:
    """
    Print detection results in a human-readable format.
    
    This function formats the detection results for console output.
    It provides clear feedback about what was detected.
    
    Args:
        result: Detection results dictionary
    
    Example output:
        Detected: Yes
        Confidence: 0.94 (94%)
        Potential Fake: No
        Auto-Tag: pothole
    """
    # Print a header
    print("\n" + "=" * 40)
    print("DETECTION RESULTS")
    print("=" * 40)
    
    # Print whether a pothole was detected
    detected_text = "Yes" if result["detected"] else "No"
    print(f"  Pothole Detected: {detected_text}")
    
    # Print confidence score (if detected)
    if result["detected"]:
        # Convert to percentage for easier reading
        confidence_pct = result["confidence"] * 100
        print(f"  Confidence: {result['confidence']:.2f} ({confidence_pct:.1f}%)")
    
    # Print potential fake status
    fake_status = "Yes" if result["is_potential_fake"] else "No"
    print(f"  Potential Fake Report: {fake_status}")
    
    # Print auto-tag
    print(f"  Auto-Tag: {result.get('auto_tag', 'N/A')}")
    
    # Print bounding box if available
    if result.get("bounding_box"):
        box = result["bounding_box"]
        print(f"  Bounding Box:")
        print(f"    - Top-Left: ({box['x1']}, {box['y1']})")
        print(f"    - Bottom-Right: ({box['x2']}, {box['y2']})")
        print(f"    - Size: {box['width']}x{box['height']} pixels")
    
    # Print message
    print(f"  Message: {result.get('message', 'N/A')}")
    
    # Print closing line
    print("=" * 40)


def save_detection_report(result: dict, image_path: str, output_path: str) -> None:
    """
    Save detection results to a text file for record keeping.
    
    This creates a simple text report that can be reviewed later.
    
    Args:
        result: Detection results dictionary
        image_path: Path to the input image
        output_path: Path where to save the report
    """
    # Get the base name of the image file
    image_name = os.path.basename(image_path)
    
    # Create the report content
    report = f"""
AyosBayan Pothole Detection Report
===================================
Image: {image_name}
Date: {os.popen('date').read().strip()}

Detection Results
-----------------
Pothole Detected: {'Yes' if result['detected'] else 'No'}
Confidence Score: {result['confidence']:.4f}
Potential Fake Report: {'Yes' if result['is_potential_fake'] else 'No'}
Auto-Tag: {result.get('auto_tag', 'N/A')}

Message: {result.get('message', 'N/A')}

Bounding Box (if detected):
"""
    
    # Add bounding box info if available
    if result.get("bounding_box"):
        box = result["bounding_box"]
        report += f"""
  - Top-Left: ({box['x1']}, {box['y1']})
  - Bottom-Right: ({box['x2']}, {box['y2']})
  - Width: {box['width']} pixels
  - Height: {box['height']} pixels
"""
    else:
        report += "  No bounding box (no detection)\n"
    
    # Write the report to file
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetection report saved to: {output_path}")


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def main(args):
    """
    Main function that runs the inference pipeline.
    
    This is the entry point for local inference. It:
    1. Loads the trained model
    2. Runs inference on the specified image
    3. Prints results
    4. Optionally saves outputs
    
    Args:
        args: Parsed command-line arguments
    """
    # Print a banner
    print("=" * 60)
    print("AyosBayan ML - Pothole Detection Inference")
    print("=" * 60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Image: {args.image}")
    print(f"  Model: {args.model}")
    print(f"  Confidence Threshold: {args.conf}")
    print(f"  Save Output: {args.save}")
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"\nERROR: Model file not found: {args.model}")
        print(f"\nTo train a model:")
        print(f"  1. Ensure dataset is prepared in data/pothole_dataset/")
        print(f"  2. Run: python src/train.py")
        print(f"  3. The model will be saved as best.pt")
        sys.exit(1)
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        print(f"\nERROR: Image file not found: {args.image}")
        print(f"\nPlease provide a valid image path.")
        sys.exit(1)
    
    # Step 1: Load the trained model
    print(f"\nLoading model: {args.model}")
    model = load_model(args.model)
    print("Model loaded successfully!")
    
    # Step 2: Run inference
    print(f"\nProcessing image: {args.image}")
    result = run_inference(model, args.image, args)
    
    # Step 3: Print results
    print_detection_result(result)
    
    # Step 4: Create API-style response
    api_response = create_api_response(result)
    
    print("\nAPI Response (JSON format):")
    print("-" * 40)
    import json
    print(json.dumps(api_response, indent=2))
    
    # Step 5: Optionally save detection report
    if args.save:
        # Create output directory if needed
        os.makedirs(args.output, exist_ok=True)
        
        # Save annotated image (if model saved one)
        # YOLO saves to output/inference/ folder
        
        # Save detection report
        report_path = os.path.join(args.output, "detection_report.txt")
        save_detection_report(result, args.image, report_path)
        
        print(f"\nOutputs saved to: {args.output}/")
    
    # Print completion
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for the inference script.
    
    When run directly, this script:
    1. Parses command-line arguments
    2. Runs the inference pipeline
    3. Handles errors gracefully
    
    Usage:
        python src/inference.py --image tests/test_images/test.jpg
        
    With all options:
        python src/inference.py --image test.jpg --model best.pt --save --conf 0.6
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Run the main inference function
        main(args)
        
    except KeyboardInterrupt:
        # Handle user pressing Ctrl+C
        print("\n\nInference interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        # Handle any other errors
        print(f"\nERROR: Inference failed with exception:")
        print(f"  {str(e)}")
        sys.exit(1)
