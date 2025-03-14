import io
from PIL import Image

from fastapi.testclient import TestClient

from mobilesam_segmentation.models import MobileSamModel
from mobilesam_segmentation.app import api

client = TestClient(api)


def test_segment_image_endpoint_success(monkeypatch):
    """
    Test that the /segment-image endpoint processes a valid image file.
    """

    # Override MobileSamModel.segment_everything to bypass heavy processing.
    def dummy_segment_everything(self, image, **kwargs):
        return image  # simply return the input image for testing

    monkeypatch.setattr(MobileSamModel, "segment_everything", dummy_segment_everything)

    # Create a dummy image file to send to the endpoint.
    image = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/segment-image", files={"file": ("test.png", buf, "image/png")})

    assert response.status_code == 200
    # Verify that the response is an image/png.
    assert response.headers["content-type"] == "image/png"


def test_segment_image_endpoint_invalid_file():
    """
    Test that the /segment-image endpoint rejects a non-image file.
    """
    response = client.post(
        "/segment-image",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )

    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]


def test_segment_image_endpoint_invalid_bbox(monkeypatch):
    """
    Test that the endpoint returns a 400 error when an invalid bbox is provided.
    """

    # Override segment_everything to avoid unnecessary processing.
    def dummy_segment_everything(self, image, **kwargs):
        return image

    monkeypatch.setattr(MobileSamModel, "segment_everything", dummy_segment_everything)

    # Create a valid dummy image.
    image = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Provide an invalid bbox string.
    response = client.post(
        "/segment-image",
        files={"file": ("test.png", buf, "image/png")},
        data={"bbox": "invalid,bbox"}
    )

    assert response.status_code == 400
    assert "Invalid bbox parameter" in response.json()["detail"]
