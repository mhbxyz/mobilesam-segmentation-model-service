import uvicorn


if __name__ == "__main__":
    uvicorn.run("mobilesam_segmentation_model_service.app:api", host="0.0.0.0", port=8000, reload=True)
