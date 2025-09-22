import napari
from magicgui import magicgui
from pathlib import Path
import os
from typing import Optional
from enum import Enum

from inference import run_i3d_inference

def get_data_layer_choices(viewer):
    """Get available image layers"""
    return [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]

@magicgui(
    call_button="Run i3d Inference",
    layout="vertical",
    model_path={"label": "Model Checkpoint", "widget_type": "FileEdit", 
                "filter": "Checkpoint (*.ckpt *.pth *.pt)"},
    layer={"label": "Input Layer", "choices": get_data_layer_choices},
    tile_size={"label": "Tile Size", "widget_type": "SpinBox", 
               "min": 32, "max": 256, "step": 32, "value": 64},
    stride={"label": "Stride (0=auto)", "widget_type": "SpinBox", 
            "min": 0, "max": 128, "step": 1, "value": 32},
    in_chans={"label": "Input Channels", "widget_type": "SpinBox", 
              "min": 10, "max": 65, "step": 1, "value": 30},
    batch_size={"label": "Batch Size", "widget_type": "SpinBox", 
                "min": 1, "max": 256, "step": 1, "value": 16},
)
def i3d_inference_widget(
    viewer: napari.Viewer,
    model_path: str,
    layer: napari.layers.Layer,
    tile_size: int = 64,
    stride: int = 32,
    in_chans: int = 30,
    batch_size: int = 16,
) -> Optional[napari.layers.Image]:
    
    if not model_path or not os.path.exists(model_path):
        print("Please select a valid model checkpoint file")
        return None
    
    if layer is None:
        print("Please select an input layer")
        return None
    
    # Auto-calculate stride if not specified (0 means auto)
    if stride == 0:
        stride = tile_size // 3
    
    try:
        results = run_i3d_inference(
            viewer=viewer,
            layer=layer,
            checkpoint_path=model_path,
            tile_size=tile_size,
            stride=stride,
            in_chans=in_chans,
            batch_size=batch_size
        )
        
        return None
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Inference error: {str(e)}")
        return None