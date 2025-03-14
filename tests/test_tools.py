import numpy as np
import torch
from PIL import Image

from fastapi.testclient import TestClient

from mobilesam_segmentation.tools import fast_process
from mobilesam_segmentation.app import api

client = TestClient(api)


def test_fast_process_default():
    """
    Test that fast_process returns an image when provided with a valid annotation.
    """
    # Create a dummy image and a dummy annotation with a segmentation mask (all zeros)
    image = Image.new("RGB", (100, 100), color="white")
    dummy_annotation = {'segmentation': np.zeros((100, 100), dtype=np.uint8)}

    result = fast_process(
        annotations=[dummy_annotation],
        image=image,
        device=torch.device("cpu"),
        scale=1,
        better_quality=False,
        mask_random_color=False,
        bbox=None,
        use_retina=False,
        with_contours=False
    )

    assert isinstance(result, Image.Image)
    # Check that the size of the result is the same as the original image
    assert result.size == image.size


def test_fast_process_with_bbox():
    """
    Test fast_process when a bbox is provided.
    """
    image = Image.new("RGB", (100, 100), color="white")
    dummy_annotation = {'segmentation': np.zeros((100, 100), dtype=np.uint8)}
    bbox = (10, 10, 50, 50)

    result = fast_process(
        annotations=[dummy_annotation],
        image=image,
        device=torch.device("cpu"),
        scale=1,
        better_quality=False,
        mask_random_color=False,
        bbox=bbox,
        use_retina=False,
        with_contours=False
    )

    assert isinstance(result, Image.Image)
