# viam-vision-prep
Viam Vision Prep (VVP): Preprocessing library for Viam Vision Services Inputs.

## **Configuring the Transform Pipeline with JSON**  

The `Transform` class allows you to define an image processing pipeline using a **JSON-like configuration**.  
This enables dynamic and flexible transformation of images **without modifying code**.  

### **JSON Configuration Format**  
The JSON configuration consists of three main fields:  
- `"device"`: Specifies whether to use `"cpu"` or `"cuda"`.  
- `"jit_script"`: Boolean indicating if the transformations should be JIT compiled.  
- `"pipeline"`: A list of transformations to apply.  

Each transformation is defined as a **dictionary** inside a **list**:  
- `"name"`: The name of the transformation (must exist in `TRANSFORM_REGISTRY`).  
- `"params"`: A dictionary of parameters for the transformation (optional).  


### **Example JSON Configurations**  

```json
{
    "device": "cuda",
    "jit_script": true,
    "pipeline": [
        {"name": "Resize", "params": {"size": [256, 256]}},
        {"name": "ToTensor"},
        {"name": "Normalize", "params": {"mean": [0.5], "std": [0.5]}}, 
        {"name": "HorizontalFlip"}
    ]
}

```

**This pipeline:**  
- **Resizes** the image to **128x128**  
- **Converts** it to a tensor  
- **Converts to grayscale**  
- **Normalizes**  
- **Applies a horizontal flip**  

---

## **How to Use JSON Config with the `Transform` Class**
### **Python Code Example**
```python
from vvp.transform import Transform
from vvp.image_object import TargetType
import torch

# Define JSON pipeline
pipeline_config = [
    {"name": "Resize", "params": {"size": [256, 256]}},
    {"name": "ToTensor"},
    {"name": "Normalize", "params": {"mean": [0.5], "std": [0.5]}},
]

# Initialize Transform class with JSON config
device = torch.device("cuda")  # Use GPU if available
transformer = Transform(target_type="np_array", config=pipeline_config)

# Apply transform
transformed_image = transformer(image_obj, TargetType.NP_ARRAY)  # Returns a NumPy array
```
---

## **Supported Transformations**
Here are the transformations you can use in your JSON config:

| Transform Name    | Description                          | Parameters |
|------------------|----------------------------------|------------|
| `Resize`         | Resizes image to a specified size | `size` (tuple or int) – Target size |
| `ToTensor`       | Converts image to a tensor       | No parameters |
| `Normalize`      | Normalizes pixel values          | `mean` (list of floats), `std` (list of floats) |
| `Grayscale`      | Converts image to grayscale      | `num_output_channels` (1 or 3, default: 1) |

