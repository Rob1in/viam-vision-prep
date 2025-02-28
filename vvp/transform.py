from typing import Any, Callable, Dict, List, Literal, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .image_object import ImageObject, TargetType

# Registry mapping transform names to their corresponding function
TRANSFORM_REGISTRY: Dict[str, Callable] = {
    "Resize": T.Resize,  # Works with both PIL and Tensors
    "ToTensor": T.ToTensor,  # Converts PIL → Tensor
    "Normalize": T.Normalize,  # Works on Tensor
    "Grayscale": T.Grayscale,  # Works on Tensor
    "HorizontalFlip": T.functional.hflip,  # Deterministic horizontal flip
    "VerticalFlip": T.functional.vflip,  # Deterministic vertical flip
}


class Transform:
    """
    Applies a list of torchvision transforms to an ImageObject and returns the transformed output.

    Args:
        target_type (str): The desired output type ("np_array", "uint8_tensor", "float32_tensor", "pil_image").
        json_config (List[Dict[str, Any]]): JSON-like list defining the transform pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        self.device = torch.device(config["device"])
        self.transform = self._build_transform_pipeline(config)  # Store pipeline

    def _build_transform_pipeline(self, config: Dict[str, Any]) -> T.Compose:
        """
        Builds a torchvision transform pipeline from a JSON configuration.
        Moves tensor-based transforms to the specified device.
        Optionally compiles the transform pipeline using torch.jit.script().
        """
        transform_list = []

        for transform_entry in config["pipeline"]:
            name = transform_entry.get("name")
            params = transform_entry.get("params", {})

            if name not in TRANSFORM_REGISTRY:
                raise ValueError(f"Unknown transform: {name}")

            transform_fn = TRANSFORM_REGISTRY[name](**params)  # Instantiate transform
            transform_list.append(transform_fn)

        transform_pipeline = T.Compose(transform_list)

        # Optionally JIT script the pipeline for optimization
        if config["jit_script"]:
            transform_pipeline = torch.jit.script(transform_pipeline)

        return transform_pipeline

    def __call__(
        self,
        image_obj: ImageObject,
        target_type: Literal["np_array", "uint8_tensor", "float32_tensor", "pil_image"],
    ) -> Union[torch.Tensor, np.ndarray, Image.Image]:
        """
        Applies the stored transform pipeline to the image and returns it in the target format.

        Args:
            target_type (str): The desired output type ("np_array", "uint8_tensor", "float32_tensor", "pil_image").

        Returns:
            - torch.Tensor if target_type is "uint8_tensor" or "float32_tensor"
            - np.ndarray if target_type is "np_array"
            - PIL.Image if target_type is "pil_image"
        """
        transformed = self.transform(image_obj.float32_tensor.to(self.device))

        if target_type == TargetType.NP_ARRAY:
            return transformed.numpy()

        elif target_type == TargetType.PIL_IMAGE:
            transformed = (
                (transformed * 255).clamp(0, 255).byte()
                if transformed.dtype == torch.float32
                else transformed
            )
            return T.ToPILImage()(transformed)

        elif target_type == TargetType.UINT8_TENSOR:
            return (
                (transformed * 255).clamp(0, 255).byte()
                if transformed.dtype == torch.float32
                else transformed
            )

        elif target_type == TargetType.FLOAT32_TENSOR:
            return transformed

        else:
            raise ValueError("Unsupported target type.")
