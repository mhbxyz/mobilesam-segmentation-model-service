from fastapi import FastAPI
from mobilesam_segmentation.routes import router

api = FastAPI(title="MobileSam Segmentation Service")

api.include_router(router)
