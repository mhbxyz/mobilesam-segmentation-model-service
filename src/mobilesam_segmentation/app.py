from contextlib import asynccontextmanager

from fastapi import FastAPI

from mobilesam_segmentation.models import MobileSamModel
from mobilesam_segmentation.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # At startup
    app.model = MobileSamModel()
    yield
    # Before shutdown


api = FastAPI(title="MobileSam Segmentation Service", lifespan=lifespan)
api.include_router(router)
