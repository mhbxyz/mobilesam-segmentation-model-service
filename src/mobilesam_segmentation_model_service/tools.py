from PIL import Image, ImageDraw
import numpy as np
import random

def fast_process(
    annotations,
    image,
    device,
    scale=1,
    better_quality=False,
    mask_random_color=False,
    bbox=None,
    use_retina=False,
    with_contours=True,
):
    """
    Process segmentation annotations and overlay them on the image.

    Parameters:
        annotations (list): List of dictionaries from MobileSam containing segmentation masks.
            Each dictionary is expected to have a key "segmentation" with a binary mask (numpy array).
        image (PIL.Image): The original input image.
        device: Computation device (not used in this implementation).
        scale (int or float): Scale factor for resizing masks if needed.
        better_quality (bool): If True, use a high-quality resampling filter when resizing masks.
        mask_random_color (bool): If True, each mask is drawn with a random color. Otherwise, a default color is used.
        bbox: Unused parameter.
        use_retina (bool): Unused parameter.
        with_contours (bool): If True, draw a bounding box (as a simple contour) around each mask.

    Returns:
        PIL.Image: The image with segmentation overlays.
    """
    # Convert the base image to RGBA to support transparency.
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Process each annotation.
    for ann in annotations:
        mask = ann.get("segmentation", None)
        if mask is None:
            continue

        # If the mask is a numpy array, convert it to a PIL Image.
        if isinstance(mask, np.ndarray):
            # Assume mask is binary (0 or 1); scale to 0-255.
            mask_img = Image.fromarray((mask * 255).astype("uint8"))
            # If a scale factor is provided, resize the mask.
            if scale != 1:
                resample_mode = Image.LANCZOS if better_quality else Image.NEAREST
                new_size = (int(mask_img.width * scale), int(mask_img.height * scale))
                mask_img = mask_img.resize(new_size, resample=resample_mode)
            # Ensure the mask matches the size of the base image.
            if mask_img.size != base.size:
                mask_img = mask_img.resize(base.size, resample=Image.NEAREST)
        else:
            continue

        # Determine the color for this mask.
        if mask_random_color:
            color = tuple(random.randint(0, 255) for _ in range(3)) + (120,)
        else:
            color = (255, 0, 0, 120)  # Default: semi-transparent red.

        # Create a colored overlay image.
        colored_mask = Image.new("RGBA", base.size, color)

        # Paste the colored mask onto the overlay using the mask image as a transparency mask.
        overlay.paste(colored_mask, (0, 0), mask_img)

        # Optionally draw a bounding box as a contour.
        if with_contours:
            bbox_coords = mask_img.getbbox()
            if bbox_coords:
                draw.rectangle(bbox_coords, outline=(0, 255, 0, 255), width=2)

    # Combine the overlay with the original image.
    combined = Image.alpha_composite(base, overlay)
    return combined.convert("RGB")
