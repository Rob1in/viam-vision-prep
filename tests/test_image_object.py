import io

import numpy as np
import pytest
from PIL import Image
from torch import Tensor
from viam.media.utils.pil import pil_to_viam_image
from viam.media.video import CameraMimeType

from vvp.image_object import ImageObject


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def sample_viam_image(sample_pil_image):
    """Convert the sample PIL image to a ViamImage."""
    return pil_to_viam_image(sample_pil_image, mime_type=CameraMimeType.JPEG)


@pytest.fixture
def sample_bytes_image(sample_pil_image):
    """Convert the sample PIL image to a byte array."""
    img_bytes = io.BytesIO()
    sample_pil_image.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


## Test instantiation from different inputs


def test_image_object_from_pil_image(sample_pil_image):
    """Test building ImageObject from a PIL image."""
    img_obj = ImageObject.from_pil_image(sample_pil_image)
    assert isinstance(img_obj, ImageObject)
    assert img_obj.pil_image is not None
    assert img_obj.pil_image.size == (100, 100)


def test_image_object_from_viam_image(sample_viam_image):
    """Test building ImageObject from a ViamImage."""
    img_obj = ImageObject.from_viam_image(sample_viam_image)
    assert isinstance(img_obj, ImageObject)
    assert img_obj.pil_image is not None
    assert img_obj.pil_image.size == (100, 100)


def test_image_object_from_bytes(sample_bytes_image):
    """Test building ImageObject from a byte array."""
    img_obj = ImageObject.from_bytes(sample_bytes_image)
    assert isinstance(img_obj, ImageObject)
    assert img_obj.pil_image is not None
    assert img_obj.pil_image.size == (100, 100)


## Test lazy evaluation pattern


@pytest.fixture
def image_object(sample_pil_image) -> ImageObject:
    """Create an ImageObject but do not access computed properties."""
    return ImageObject.from_pil_image(sample_pil_image)


def test_lazy_np_array_initially_none(image_object):
    """Test that _np_array is None at initialization."""
    assert image_object._np_array is None


def test_lazy_np_array_computed_on_access(image_object):
    """Test that _np_array is computed only when accessed."""
    _ = image_object.np_array  # Accessing should trigger computation
    assert image_object._np_array is not None
    assert isinstance(image_object.np_array, np.ndarray)


def test_lazy_uint8_tensor_initially_none(image_object):
    """Test that _uint8_tensor is None at initialization."""
    assert image_object._uint8_tensor is None


def test_lazy_uint8_tensor_computed_on_access(image_object):
    """Test that _uint8_tensor is computed only when accessed."""
    _ = image_object.uint8_tensor  # Trigger computation
    assert image_object._uint8_tensor is not None
    assert isinstance(image_object.uint8_tensor, Tensor)


def test_lazy_float32_tensor_initially_none(image_object):
    """Test that _float32_tensor is None at initialization."""
    assert image_object._float32_tensor is None


def test_lazy_float32_tensor_computed_on_access(image_object):
    """Test that _float32_tensor is computed only when accessed."""
    _ = image_object.float32_tensor  # Trigger computation
    assert image_object._float32_tensor is not None
    assert isinstance(image_object.float32_tensor, Tensor)


if __name__ == "__main__":
    pytest.main()
