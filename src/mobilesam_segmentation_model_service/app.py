from fastapi import FastAPI
from mobilesam_segmentation_model_service.routes import router

api = FastAPI(title="MobileSam Segmentation Service")

api.include_router(router)
