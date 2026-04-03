"""
AyosBayan ML Module - FastAPI Server
=====================================

This file contains the FastAPI server for the pothole detection API.
It provides a REST API endpoint that accepts image uploads and returns
detection results.

This is the production backend that powers the AyosBayan web platform.
When a citizen uploads a photo of a road problem, the backend sends
the photo to this API, which runs the YOLOv8 model and returns the
detection results.

Author: Federex (AyosBayan Research Project)
Version: 1.0
Last Updated: April 2026

How to Run:
    uvicorn src.api:app --reload --port 8000
    
API Endpoint:
    POST /detect-pothole
    
Request:
    Upload an image file (JPEG, PNG, etc.)
    
Response (JSON):
    {
        "detected": true,
        "confidence": 0.94,
        "is_potential_fake": false,
        "auto_tag": "pothole",
        "message": "Pothole detected with high confidence"
    }

API Documentation:
    Once running, visit http://localhost:8000/docs for interactive API docs
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

# FastAPI - The web framework for building the API
# FastAPI is:
#   - Fast (high performance, similar to Node.js/Go)
#   - Easy to use (Python-native)
#   - Auto-generates API documentation
#   - Provides automatic request validation
from fastapi import FastAPI, File, UploadFile, HTTPException

# Uvicorn - The ASGI server that runs FastAPI
# This is the server that actually handles HTTP requests
# We'll run this via command line: uvicorn src.api:app

# Import for response models (Pydantic)
# Pydantic is used for data validation and serialization
from pydantic import BaseModel, Field

# Import for handling uploaded files
# FastAPI uses Starlette for file handling
from typing import Optional

# Import for temporary file handling
# We'll save uploaded files temporarily for inference
import tempfile

# Import for file operations
import os

# Import for JSON serialization
import json

# Import our utility functions from utils.py
# These handle the core ML operations:
# - Model loading
# - Image validation
# - Result processing
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
# API CONFIGURATION
# ============================================================================

# Create the FastAPI application instance
# 
# FastAPI apps are instances of the FastAPI class.
# We configure it with:
#   - title: The name shown in API docs
#   - description: What the API does
#   - version: API version number
#   - docs_url: Where to find Swagger UI documentation
#   - redoc_url: Where to find ReDoc documentation
app = FastAPI(
    title="AyosBayan Pothole Detection API",
    description="""
    ## AyosBayan ML Module - Automated Pothole Detection
    
    This API provides real-time pothole detection for citizen infrastructure
    reporting. When a citizen uploads a photo of a road problem through the
    AyosBayan web platform, this API automatically detects whether the image
    contains a pothole.
    
    ### Features
    
    - **Real-time detection**: Processes images in < 2 seconds
    - **Confidence scoring**: Returns confidence scores (0.0 - 1.0)
    - **Fake report detection**: Flags suspicious reports for manual review
    - **Bounding boxes**: Returns coordinates for visualization
    
    ### Use Cases
    
    1. **Citizen Report Verification**: Auto-verify pothole reports
    2. **LGU Dashboard**: Pre-classified reports for administrators
    3. **Quality Control**: Identify potential fake reports
    
    ### Response Codes
    
    - `200`: Success - Detection completed
    - `400`: Bad Request - Invalid image format
    - `404`: Not Found - Model not loaded
    - `500`: Internal Server Error - Processing failed
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# RESPONSE MODELS
# ============================================================================

# These Pydantic models define the structure of API responses.
# They provide:
# - Data validation (ensures correct types)
# - Documentation (auto-generated in Swagger UI)
# - Serialization (convert to/from JSON)


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for the detected pothole.
    
    This model represents the location of a detected pothole
    in the image, expressed as pixel coordinates.
    """
    x1: int = Field(..., description="X coordinate of top-left corner")
    y1: int = Field(..., description="Y coordinate of top-left corner")
    x2: int = Field(..., description="X coordinate of bottom-right corner")
    y2: int = Field(..., description="Y coordinate of bottom-right corner")
    width: int = Field(..., description="Width of bounding box in pixels")
    height: int = Field(..., description="Height of bounding box in pixels")


class DetectionResponse(BaseModel):
    """
    Response model for the /detect-pothole endpoint.
    
    This is the main API response that contains all the
    detection results. It's returned as JSON to the client.
    """
    detected: bool = Field(
        ...,
        description="Whether a pothole was detected in the image"
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    is_potential_fake: bool = Field(
        ...,
        description="Whether the report needs manual review"
    )
    auto_tag: str = Field(
        ...,
        description="Auto-assigned tag ('pothole' or 'normal')"
    )
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Bounding box coordinates (only if detected)"
    )


class HealthResponse(BaseModel):
    """
    Response model for the /health endpoint.
    
    Simple health check response to verify the API is running.
    """
    status: str = Field(..., description="API status (healthy/degraded)")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: str = Field(..., description="Path to the model file")
    confidence_threshold: float = Field(
        ...,
        description="Current confidence threshold"
    )


class ModelInfoResponse(BaseModel):
    """
    Response model for the /model-info endpoint.
    
    Returns information about the loaded model.
    """
    model_type: str = Field(..., description="Type of YOLOv8 model")
    num_classes: int = Field(..., description="Number of detectable classes")
    class_names: list = Field(..., description="List of class names")
    confidence_threshold: float = Field(..., description="Current threshold")
    input_image_size: int = Field(..., description="Model input size")


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Global model variable
# We load the model once at startup and keep it in memory
# This avoids the overhead of loading the model for each request
# 
# Why global?
# - Loading the model takes several seconds
# - We want to keep it ready for fast inference
# - Multiple requests can share the same model instance
model = None

# Model loaded flag
# Tracks whether the model has been successfully loaded
model_loaded = False


# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    This function is called when the FastAPI server starts.
    We use it to:
    1. Load the YOLOv8 model into memory
    2. Prepare the model for inference
    
    This ensures the model is ready before handling requests.
    """
    global model, model_loaded
    
    # Print startup message
    print("=" * 60)
    print("AyosBayan Pothole Detection API - Starting Up")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"\nWARNING: Model file not found at {DEFAULT_MODEL_PATH}")
        print("The model will be loaded when first request arrives.")
        print(f"\nTo train a model, run: python src/train.py")
        model_loaded = False
        return
    
    # Load the model
    print(f"\nLoading model: {DEFAULT_MODEL_PATH}")
    try:
        model = load_model(DEFAULT_MODEL_PATH)
        model_loaded = True
        print("Model loaded successfully!")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {str(e)}")
        model_loaded = False
    
    print("\n" + "=" * 60)
    print("API Ready!")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    This function is called when the FastAPI server stops.
    We use it to:
    1. Clean up any resources
    2. Print shutdown message
    
    Note: The model will be automatically garbage collected.
    """
    print("\n" + "=" * 60)
    print("AyosBayan Pothole Detection API - Shutting Down")
    print("=" * 60)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information.
    
    Returns basic information about the API.
    
    Returns:
        dict: Welcome message and API info
    """
    return {
        "message": "Welcome to AyosBayan Pothole Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "detect": "/detect-pothole",
            "health": "/health",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and whether the model is loaded.
    This is useful for:
    - Load balancer health checks
    - Monitoring
    - Debugging
    
    Returns:
        HealthResponse: Health status information
    """
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": DEFAULT_MODEL_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model information endpoint.
    
    Returns detailed information about the loaded model.
    This is useful for:
    - API documentation
    - Debugging
    - Client-side configuration
    
    Returns:
        ModelInfoResponse: Model information
    
    Raises:
        HTTPException: 404 if model is not loaded
    """
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=404,
            detail="Model not loaded. Please ensure best.pt exists."
        )
    
    # Return model information
    return {
        "model_type": "YOLOv8 (nano)",
        "num_classes": 1,
        "class_names": ["pothole"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "input_image_size": 640
    }


@app.post("/detect-pothole", response_model=DetectionResponse)
async def detect_pothole(file: UploadFile = File(...)):
    """
    Detect pothole in an uploaded image.
    
    This is the main API endpoint. It:
    1. Accepts an uploaded image file
    2. Validates the image format
    3. Runs the YOLOv8 model for inference
    4. Processes and returns the detection results
    
    The endpoint accepts image files in common formats:
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    - BMP (.bmp)
    - WebP (.webp)
    - TIFF (.tiff)
    
    The response includes:
    - detected: Whether a pothole was found
    - confidence: How confident the model is (0.0 - 1.0)
    - is_potential_fake: Whether the report needs manual review
    - auto_tag: Classification tag
    - message: Human-readable status
    - bounding_box: Location of pothole in image (if detected)
    
    Args:
        file: Uploaded image file (from form data)
    
    Returns:
        DetectionResponse: Detection results
    
    Raises:
        HTTPException: 400 if image is invalid
        HTTPException: 404 if model not loaded
        HTTPException: 500 if processing fails
    
    Example Usage:
        ```python
        import requests
        
        # Upload an image for detection
        with open("pothole.jpg", "rb") as f:
            response = requests.post(
                "http://localhost:8000/detect-pothole",
                files={"file": f}
            )
        
        result = response.json()
        print(f"Detected: {result['detected']}")
        print(f"Confidence: {result['confidence']}")
        ```
    """
    global model, model_loaded
    
    # Step 1: Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model not loaded. Please ensure best.pt exists in the project "
                "root directory. Run python src/train.py to train the model."
            )
        )
    
    # Step 2: Validate the uploaded file
    # 
    # We need to check:
    # 1. File was actually uploaded
    # 2. File has a valid image extension
    # 3. File is not empty
    
    # Check if file exists
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded. Please provide an image file."
        )
    
    # Get the filename from the upload
    # This helps with error messages and logging
    filename = file.filename
    
    # Check file extension
    # FastAPI doesn't automatically validate extensions
    # We do it manually to provide clear error messages
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file format: {filename}. "
                f"Supported formats: JPEG, PNG, BMP, WebP, TIFF"
            )
        )
    
    # Step 3: Save the uploaded file temporarily
    # 
    # YOLO needs a file path, not the uploaded file object.
    # We save the uploaded file to a temporary location,
    # run inference, then delete it.
    #
    # Using tempfile:
    # - Creates a temporary file in the OS temp directory
    # - Automatically handles file naming
    # - Will be cleaned up when the context exits
    
    # Create a temporary file with the same extension
    # We use the original filename extension to maintain format
    file_extension = os.path.splitext(filename)[1]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:
        # Read the uploaded file content
        # file.file is a file-like object containing the uploaded data
        content = await file.read()
        
        # Write to temporary file
        temp_file.write(content)
        
        # Get the temporary file path
        temp_file_path = temp_file.name
    
    try:
        # Step 4: Validate the image
        # 
        # We use our utility function to check if the image
        # is valid and can be processed. This catches:
        # - Corrupted image files
        # - Files that aren't valid images
        # - Files that are too small or empty
        is_valid, message = validate_image(temp_file_path)
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image: {message}"
            )
        
        # Step 5: Run inference
        # 
        # This is where the actual ML prediction happens.
        # The YOLO model:
        # - Reads the image from temp_file_path
        # - Preprocesses it (resizes to 640x640)
        # - Runs the neural network
        # - Returns detection results
        #
        # This typically takes < 2 seconds on CPU
        print(f"Running inference on: {filename}")
        
        # Run the prediction
        # results is a list (one entry per image)
        results = model.predict(
            source=temp_file_path,    # Input image path
            conf=CONFIDENCE_THRESHOLD,  # Confidence threshold
            iou=0.7,                   # IOU threshold for NMS
            verbose=False,            # Don't print verbose output
            device='',                # Auto-detect (cpu or cuda)
            show=False,               # Don't display window
            save=False,               # Don't save annotated image
        )
        
        # Step 6: Process the results
        # 
        # The raw results from YOLO need to be converted to
        # our standardized format. We use our utility function.
        detection_result = process_detection_result(results)
        
        # Step 7: Create API response
        # 
        # Convert the detection result to our API response format
        # This adds:
        # - Human-readable message
        # - Auto-tag (pothole or normal)
        # - Clean JSON structure
        api_response = create_api_response(detection_result)
        
        # Return the response
        # FastAPI will automatically serialize this to JSON
        print(f"Detection complete: {api_response['detected']}")
        
        return api_response
        
    except HTTPException:
        # Re-raise HTTP exceptions (don't wrap them)
        raise
        
    except Exception as e:
        # Handle any unexpected errors
        # This catches:
        # - YOLO processing errors
        # - File I/O errors
        # - Memory errors
        print(f"ERROR: Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )
        
    finally:
        # Step 8: Clean up temporary file
        # 
        # Always delete the temporary file, even if an error occurred.
        # This prevents accumulation of temp files on the server.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Custom 404 error handler.
    
    Returns a JSON response instead of the default HTML page.
    """
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Custom 500 error handler.
    
    Returns a JSON response for internal server errors.
    """
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "detail": str(exc)
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    This block allows running the API directly with Python.
    
    Normally, you'd run with: uvicorn src.api:app --reload --port 8000
    
    But you can also run directly:
        python src/api.py
    
    This is mainly useful for development/debugging.
    """
    import uvicorn
    
    print("Starting AyosBayan Pothole Detection API...")
    print("Visit http://localhost:8000/docs for API documentation")
    
    # Run the server
    uvicorn.run(
        "src.api:app",     # The app module and instance
        host="0.0.0.0",    # Listen on all interfaces
        port=8000,         # Port number
        reload=True        # Auto-reload on code changes
    )
