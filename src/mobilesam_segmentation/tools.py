from PIL import Image, ImageDraw
import numpy as np
import random
import torch
import torch.nn.functional as F


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
        device: Computation device. If it is a GPU device, mask resizing will use torch on that device.
        scale (int or float): Scale factor for resizing masks if needed.
        better_quality (bool): If True, use a high-quality resampling filter when resizing masks.
        mask_random_color (bool): If True, each mask is drawn with a random color. Otherwise, a default color is used.
        bbox: If provided as a tuple (left, upper, right, lower), this bounding box is drawn on the final image.
        use_retina (bool): If True, the final image is upscaled (e.g., doubled in size) for retina displays.
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

        # Process mask using torch on GPU if device is available and not CPU.
        if isinstance(mask, np.ndarray):
            if device and device.type != "cpu":
                # Convert mask to torch tensor, add batch and channel dimensions.
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                # Resize mask using torch interpolation if scale is different from 1.
                if scale != 1:
                    new_size = (int(mask_tensor.shape[-2] * scale), int(mask_tensor.shape[-1] * scale))
                    mode = "bilinear" if better_quality else "nearest"
                    mask_tensor = F.interpolate(mask_tensor, size=new_size, mode=mode, align_corners=False if mode=="bilinear" else None)
                # Ensure the mask matches the size of the base image.
                if (mask_tensor.shape[-2], mask_tensor.shape[-1]) != (base.height, base.width):
                    mask_tensor = F.interpolate(mask_tensor, size=(base.height, base.width), mode="nearest")
                # Convert mask back to PIL Image.
                mask_tensor = (mask_tensor.squeeze() * 255).clamp(0, 255).byte().cpu().numpy()
                mask_img = Image.fromarray(mask_tensor)
            else:
                # Fallback: Use PIL for resizing.
                mask_img = Image.fromarray((mask * 255).astype("uint8"))
                if scale != 1:
                    resample_mode = Image.LANCZOS if better_quality else Image.NEAREST
                    new_size = (int(mask_img.width * scale), int(mask_img.height * scale))
                    mask_img = mask_img.resize(new_size, resample=resample_mode)
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

        # Optionally draw a bounding box around each mask.
        if with_contours:
            mask_bbox = mask_img.getbbox()
            if mask_bbox:
                draw.rectangle(mask_bbox, outline=(0, 255, 0, 255), width=2)

    # If a bbox parameter is provided, draw it on the overlay (in blue).
    if bbox is not None:
        # bbox is assumed to be a tuple: (left, upper, right, lower)
        draw.rectangle(bbox, outline=(0, 0, 255, 255), width=2)

    # Combine the overlay with the original image.
    combined = Image.alpha_composite(base, overlay)
    result = combined.convert("RGB")

    # If use_retina is True, upscale the result for better display on high-DPI screens.
    if use_retina:
        new_size = (result.width * 2, result.height * 2)
        result = result.resize(new_size, resample=Image.LANCZOS)

    return result
