# ============================================================================
# AyosBayan Pothole Detection - Dockerfile
# ============================================================================
#
# This Dockerfile creates a Docker container for the AyosBayan pothole
# detection API. It includes:
#   - Python base image
#   - Required dependencies
#   - The trained model (best.pt)
#   - The FastAPI application
#
# Usage:
#   1. Build the image:
#      docker build -t ayosbayan-pothole-api .
#
#   2. Run the container:
#      docker run -p 8000:8000 ayosbayan-pothole-api
#
#   3. Test the API:
#      curl -X POST -F "file=@test.jpg" http://localhost:8000/detect-pothole
#
# Author: Federex (AyosBayan Research Project)
# Version: 1.0
# Last Updated: April 2026
# ============================================================================

# ============================================================================
# BASE IMAGE
# ============================================================================

# Use Python 3.10 as the base image
# This is the version specified in the README
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Set Python to not cache packages
ENV PYTHONDONTWRITEBYTECODE=1

# Set Python to use unbuffered output (logs appear immediately)
ENV PYTHONUNBUFFERED=1


# ============================================================================
# INSTALL SYSTEM DEPENDENCIES
# ============================================================================

# Install system dependencies needed for:
# - libGL: OpenGL library (required by some image processing libraries)
# - libglib2.0-0: GLib library (required by OpenCV)
# These are needed for Pillow and Ultralytics to process images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================================
# COPY REQUIREMENTS
# ============================================================================

# Copy only the requirements file first
# This allows Docker to cache the dependency installation
COPY requirements.txt /app/requirements.txt


# ============================================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================================

# Install Python dependencies from requirements.txt
# Using pip install with no cache to keep image small
RUN pip install --no-cache-dir -r /app/requirements.txt


# ============================================================================
# COPY APPLICATION FILES
# ============================================================================

# Copy the source code to the container
# We copy in the following order (from least to most frequently changing):
# 1. src/ (application code)
# 2. tests/ (test images)
# 3. data/ (dataset configuration)
# This order helps with Docker layer caching

# Copy the src directory
COPY src/ /app/src/

# Copy tests directory (test images)
COPY tests/ /app/tests/

# Copy data directory (data.yaml)
COPY data/ /app/data/

# Copy the trained model (best.pt)
# Note: You need to place best.pt in the project root before building
# The model is not included in the repository (see .dockerignore)
# COPY best.pt /app/best.pt  # Uncomment after training


# ============================================================================
# EXPOSE PORT
# ============================================================================

# Expose port 8000 for the FastAPI server
# This is the default port for Uvicorn
EXPOSE 8000


# ============================================================================
# HEALTH CHECK
# ============================================================================

# Add a health check to verify the API is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# ============================================================================
# RUN COMMAND
# ============================================================================

# Run the FastAPI server using Uvicorn
# 
# Command breakdown:
#   - uvicorn: The ASGI server
#   - src.api:app: The FastAPI application (src/api.py, app variable)
#   - --host 0.0.0.0: Listen on all network interfaces
#   - --port 8000: Listen on port 8000
#   - --reload: Auto-reload on code changes (dev mode only)
#
# For production, remove --reload:
#   CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
