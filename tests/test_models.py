import numpy as np
from PIL import Image

from fastapi.testclient import TestClient

from mobilesam_segmentation.models import MobileSamModel
from mobilesam_segmentation.app import api

client = TestClient(api)


def test_segment_everything(monkeypatch):
    """
    Test MobileSamModel.segment_everything by replacing the mask generator with a dummy function.
    """
    # Create a dummy image
    image = Image.new("RGB", (200, 200), color="white")
    model = MobileSamModel()

    # Define a dummy generate function that returns a list with a dummy annotation.
    def dummy_generate(nd_image):
        # Create a dummy segmentation mask (all ones) with the same dimensions as the numpy array image.
        return [{'segmentation': np.ones(nd_image.shape[:2], dtype=np.uint8)}]

    # Monkeypatch the mask generator’s generate method.
    monkeypatch.setattr(model.mask_generator, "generate", dummy_generate)

    # Call segment_everything with controlled parameters.
    result_image = model.segment_everything(
        image,
        input_size=200,
        better_quality=False,
        with_contours=False,
        use_retina=False,
        mask_random_color=False,
        bbox=None
    )

    assert isinstance(result_image, Image.Image)
