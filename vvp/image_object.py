import io
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from viam.media.video import ViamImage  # Assuming this is the correct import


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

    def __init__(self, pil_image=None, device: Optional[str] = None):
        """
        Private constructor. Use factory methods to create instances.
        """
        self._pil_image = pil_image
        self._np_array = None
        self._uint8_tensor = None
        self._float32_tensor = None
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @classmethod
    def from_viam_image(cls, viam_image: ViamImage, device: Optional[str] = None):
        """Creates an ImageObject from a ViamImage."""
        pil_image = Image.open(io.BytesIO(viam_image.data)).convert("RGB")
        return cls(pil_image=pil_image, device=device)

    @classmethod
    def from_pil_image(cls, pil_image: Image, device: Optional[str] = None):
        """Creates an ImageObject from a PIL Image."""
        return cls(pil_image=pil_image, device=device)

    @classmethod
    def from_bytes(cls, image_bytes: bytes, device: Optional[str] = None):
        """Creates an ImageObject from raw image bytes."""
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cls(pil_image=pil_image, device=device)

    @property
    def pil_image(self) -> Image:
        """Returns the PIL image."""
        return self._pil_image

    @property
    def np_array(self) -> np.ndarray:
        """Lazily computes and returns the NumPy array representation of the image."""
        if self._np_array is None:
            self._np_array = np.array(self._pil_image, dtype=np.uint8)
        return self._np_array

    @property
    def uint8_tensor(self) -> torch.Tensor:
        """Lazily computes and returns the uint8 tensor representation of the image."""
        if self._uint8_tensor is None:
            self._uint8_tensor = get_tensor_from_np_array(self.np_array, "uint8").to(
                self.device
            )
        return self._uint8_tensor

    @property
    def float32_tensor(self) -> torch.Tensor:
        """Lazily computes and returns the float32 tensor representation of the image."""
        if self._float32_tensor is None:
            self._float32_tensor = get_tensor_from_np_array(
                self.np_array, "float32"
            ).to(self.device)
        return self._float32_tensor
