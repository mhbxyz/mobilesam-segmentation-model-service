# MobileSam Segmentation Model Service

## Overview

The MobileSam Segmentation Model Service is a FastAPI-based API that uses the MobileSAM model for image segmentation. It processes input images and overlays segmentation masks with options to improve quality, add contours, or apply retina upscaling.

## Features

- **Image Segmentation:** Automatically segments images using MobileSAM.
- **Customizable Processing:** Options include quality enhancement, bounding box overlays, and random mask colors.
- **API Endpoint:** A single endpoint (`/segment-image`) to submit images and receive segmented outputs.
- **Interactive Documentation:** Automatically generated API docs available via FastAPI.

## Prerequisites

- **Python:** 3.10 or later.
- **Optional:** A CUDA-enabled GPU for accelerated processing (if available).

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mhbxyz/mobilesam-segmentation-model-service
   cd mobilesam-segmentation-model-service
   ```
   
2. **Install `uv` Package Manager**  
    You can find all the instructions here: https://github.com/astral-sh/uv

3. **Create a Virtual Environment and Install Dependencies**
   ```bash
   uv sync
   ```
   This command creates the virtual environment, reads the pyproject.toml and installs all required dependencies including FastAPI, MobileSAM, Torch, and others.

## Running the Service

You can start the service in one of two ways:

### Using Uvicorn

Run the API with Uvicorn:
```bash
uv run uvicorn mobilesam_segmentation.app:api --host 0.0.0.0 --port 8000
```

### Using the Main Module

Alternatively, run the main module:
```bash
uv run python -m mobilesam_segmentation
```
After starting the server, the API will be available at http://localhost:8000.

## API Usage

**Endpoint:** `/segment-image`  
- **Method:** `POST`
- **Description:** Accepts an image file, applies segmentation using MobileSAM, and returns a processed image in PNG format. 
- **Parameters:**
  - **file** (required): The image file to process. Must be a valid image (e.g., PNG, JPEG). 
  - **input_size** (optional): Integer value (default is 1024) that controls the input image size. 
  - **better_quality** (optional): Boolean flag (default is False) to enable higher-quality mask resizing. 
  - **with_contours** (optional): Boolean flag (default is True) to draw contours around segmentation masks. 
  - **use_retina** (optional): Boolean flag (default is True) to upscale the result for high-DPI displays. 
  - **mask_random_color** (optional): Boolean flag (default is True) to use random colors for the masks. 
  - **bbox** (optional): A string representing a bounding box in the format `"left,upper,right,lower"` (e.g., `"10,10,50,50"`). If provided, this box is drawn on the final image.
- **Response:**
  - Returns the segmented image with a Content-Type of image/png.
  
### Example with `curl`
```bash
curl -X POST "http://localhost:8000/segment-image" \
  -F "file=@path_to_your_image.png" \
  -F "bbox=10,10,50,50"
```
### Interactive API Documentation

Visit http://localhost:8000/docs in your browser for an interactive API interface provided by FastAPI.

## Running Tests

The project includes a suite of tests to verify functionality. To run the tests, execute:
```bash
uv run pytest
```
This command will run the tests defined in `test_tools.py`, `test_models.py`, and `test_routes.py`.

## Docker Deployment

If you prefer to run the service in a Docker container, follow these steps:

1. **Build the Docker Image**
   ```bash
   docker build -t mobilesam-segmentation .
   ```
2. **Run the Docker Container**
    ```bash
   docker run --cpus=2 --memory=4g -p 8000:8000 mobilesam-segmentation
   ```
   The API will be accessible at http://localhost:8000 once the container is running.
