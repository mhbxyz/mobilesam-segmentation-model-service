import io

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from mobilesam_segmentation.models import MobileSamModel

router = APIRouter()


@router.post("/segment-image")
async def segment_image(
    file: UploadFile = File(...),
    input_size: int = 1024,
    better_quality: bool = False,
    with_contours: bool = True,
    use_retina: bool = True,
    mask_random_color: bool = True,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing image")

    segmented_image = MobileSamModel().segment_everything(
        image,
        input_size=input_size,
        better_quality=better_quality,
        with_contours=with_contours,
        use_retina=use_retina,
        mask_random_color=mask_random_color,
    )

    buf = io.BytesIO()
    segmented_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
