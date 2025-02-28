import numpy as np
import pytest
import torch
from PIL import Image

from vvp.image_object import ImageObject, TargetType
from vvp.transform import Transform


@pytest.fixture
def sample_image():
    """Creates a sample image object for testing."""
    width, height = 128, 128
    pil_image = Image.new("RGB", (width, height), color="white")
    return ImageObject(pil_image)


@pytest.fixture
def transform_config():
    """Returns a sample transform configuration."""
    return {
        "device": "cpu",
        "jit_script": False,
        "pipeline": [
            {"name": "Resize", "params": {"size": [64, 64]}},
            {"name": "Normalize", "params": {"mean": [0.5], "std": [0.5]}},
        ],
    }


def test_transform_float32_tensor(sample_image, transform_config):
    """Tests transformation when target_type is float32_tensor."""
    transform = Transform(transform_config)
    transformed_output = transform(sample_image, TargetType.FLOAT32_TENSOR)

    assert isinstance(transformed_output, torch.Tensor)
    assert transformed_output.dtype == torch.float32
    assert transformed_output.shape[1:] == (64, 64)  # Check resizing


def test_transform_uint8_tensor(sample_image, transform_config):
    """Tests transformation when target_type is uint8_tensor."""
    transform = Transform(transform_config)
    transformed_output = transform(sample_image, TargetType.UINT8_TENSOR)

    assert isinstance(transformed_output, torch.Tensor)
    assert transformed_output.dtype == torch.uint8
    assert transformed_output.shape[1:] == (64, 64)


def test_transform_np_array(sample_image, transform_config):
    """Tests transformation when target_type is np_array."""
    transform = Transform(transform_config)
    transformed_output = transform(sample_image, TargetType.NP_ARRAY)

    assert isinstance(transformed_output, np.ndarray)
    assert transformed_output.shape[1:] == (64, 64)


def test_transform_pil_image(sample_image, transform_config):
    """Tests transformation when target_type is pil_image."""
    transform = Transform(transform_config)
    transformed_output = transform(sample_image, TargetType.PIL_IMAGE)

    assert isinstance(transformed_output, Image.Image)
    assert transformed_output.size == (64, 64)


def test_invalid_target_type(sample_image, transform_config):
    """Ensures an invalid target type raises an error."""
    transform = Transform(transform_config)
    with pytest.raises(ValueError):
        transform(sample_image, "invalid_type")


def test_invalid_transform_name(sample_image):
    """Ensures an unknown transform name raises an error."""
    invalid_config = {
        "device": "cpu",
        "jit_script": False,
        "pipeline": [{"name": "NonExistentTransform"}],
    }

    with pytest.raises(ValueError):
        Transform(invalid_config)
