import io
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from viam.media.video import ViamImage  # Assuming this is the correct import


class TargetType:
    """Defines the possible image representations within an ImageObject."""

    NP_ARRAY = "np_array"
    UINT8_TENSOR = "uint8_tensor"
    FLOAT32_TENSOR = "float32_tensor"
    PIL_IMAGE = "pil_image"

    @classmethod
    def valid_types(cls):
        """Returns a set of valid input types."""
        return {cls.NP_ARRAY, cls.UINT8_TENSOR, cls.FLOAT32_TENSOR}


def get_tensor_from_np_array(
    np_array: np.ndarray, dtype: Literal["uint8", "float32"]
) -> torch.Tensor:
    """
    Converts a NumPy array into a PyTorch tensor.

    Args:
        np_array (np.ndarray): The input NumPy array (H, W, C).
        dtype (str): The desired data type ("uint8" or "float32").

    Returns:
        torch.Tensor: The converted tensor with shape (C, H, W).
    """
    tensor = (
        torch.from_numpy(np_array).permute(2, 0, 1).contiguous()
    )  # Convert to (C, H, W)

    if dtype == "float32":
        return tensor.to(dtype=torch.float32)
    elif dtype == "uint8":
        return tensor.to(dtype=torch.uint8)
    else:
        raise ValueError("Invalid dtype. Choose either 'uint8' or 'float32'.")


class ImageObject:
    """
    ImageObject is a wrapper around an image, supporting lazy evaluation and GPU acceleration.
    It allows initialization from different sources such as ViamImage, PIL Image, or raw bytes.
    """

    def __init__(
        self, pil_image: Optional[Image.Image] = None, device: Optional[str] = None
    ):
        """
        Private constructor. Use factory methods to create instances.
        """
        self._pil_image = pil_image
        self._cached_values = {  # Dictionary to hold lazy-loaded attributes
            TargetType.NP_ARRAY: None,
            TargetType.UINT8_TENSOR: None,
            TargetType.FLOAT32_TENSOR: None,
        }
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @classmethod
    def from_viam_image(cls, viam_image: ViamImage, device: Optional[str] = None):
        """Creates an ImageObject from a ViamImage."""
        pil_image = Image.open(io.BytesIO(viam_image.data)).convert("RGB")
        return cls(pil_image=pil_image, device=device)

    @classmethod
    def from_pil_image(cls, pil_image: Image.Image, device: Optional[str] = None):
        """Creates an ImageObject from a PIL Image."""
        return cls(pil_image=pil_image, device=device)

    @classmethod
    def from_bytes(cls, image_bytes: bytes, device: Optional[str] = None):
        """Creates an ImageObject from raw image bytes."""
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cls(pil_image=pil_image, device=device)

    @property
    def pil_image(self) -> Image.Image:
        """Returns the PIL image."""
        return self._pil_image

    def get(self, attr: str):
        """
        Generic lazy-evaluation getter for image representations. PIL Image should always be available.

        Args:
            attr (str): One of "np_array", "uint8_tensor", or "float32_tensor".

        Returns:
            The computed or cached image representation.
        """
        if attr not in self._cached_values:
            raise ValueError(
                f"Invalid attribute '{attr}'. Choose from: {list(self._cached_values.keys())}"
            )

        if self._cached_values[attr] is None:
            if attr == TargetType.NP_ARRAY:
                self._cached_values[attr] = np.array(self._pil_image, dtype=np.uint8)
            elif attr == TargetType.UINT8_TENSOR:
                self._cached_values[attr] = get_tensor_from_np_array(
                    self.get(TargetType.NP_ARRAY), "uint8"
                ).to(self.device)
            elif attr == TargetType.FLOAT32_TENSOR:
                self._cached_values[attr] = get_tensor_from_np_array(
                    self.get(TargetType.NP_ARRAY), "float32"
                ).to(self.device)

        return self._cached_values[attr]

    @property
    def np_array(self) -> np.ndarray:
        """Returns the NumPy array representation of the image, computed lazily."""
        return self.get(TargetType.NP_ARRAY)

    @property
    def uint8_tensor(self) -> torch.Tensor:
        """Returns the uint8 tensor representation of the image, computed lazily."""
        return self.get(TargetType.UINT8_TENSOR)

    @property
    def float32_tensor(self) -> torch.Tensor:
        """Returns the float32 tensor representation of the image, computed lazily."""
        return self.get(TargetType.FLOAT32_TENSOR)
