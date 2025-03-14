import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from mobilesam_segmentation_model_service.tools import fast_process


class MobileSamModel:
    def __init__(self, checkpoint: str = "resources/mobile_sam.pt", model_type: str = "vit_t"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mobile_sam = self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.mobile_sam)
        self.predictor = SamPredictor(self.mobile_sam)

    @torch.no_grad()
    def segment_everything(
            self,
            image: Image.Image,
            input_size: int = 1024,
            better_quality: bool = False,
            with_contours: bool = True,
            use_retina: bool = True,
            mask_random_color: bool = True
    ):
        input_size = int(input_size)
        w, h = image.size
        scale = input_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h))
        nd_image = np.array(image)
        annotations = self.mask_generator.generate(nd_image)
        fig = fast_process(
            annotations=annotations,
            image=image,
            device=self.device,
            scale=(1024 // input_size),
            better_quality=better_quality,
            mask_random_color=mask_random_color,
            bbox=None,
            use_retina=use_retina,
            with_contours=with_contours,
        )
        return image
