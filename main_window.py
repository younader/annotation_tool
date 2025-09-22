import napari
from magicgui import magicgui, widgets
from pathlib import Path
import numpy as np

from inference_widget import i3d_inference_widget
from training import train_i3d_model

@magicgui(
    call_button='Train i3d Model',
    layout="vertical",
    max_epochs={'widget_type': 'SpinBox', 'label': 'Max Epochs', 
                'min': 1, 'max': 100, 'value': 10},
    batch_size={'widget_type': 'SpinBox', 'label': 'Batch Size', 
                'min': 1, 'max': 128, 'value': 32},
    learning_rate={'widget_type': 'FloatSpinBox', 'label': 'Learning Rate', 
                   'min': 1e-6, 'max': 1e-2, 'step': 1e-6, 'value': 2e-5},
    tile_size={'widget_type': 'SpinBox', 'label': 'Tile Size', 
               'min': 32, 'max': 256, 'step': 32, 'value': 64},
    in_chans={'widget_type': 'SpinBox', 'label': 'Input Channels', 
              'min': 10, 'max': 65, 'step': 1, 'value': 30},
    checkpoint_dir={'widget_type': 'FileEdit', 'label': 'Checkpoint Directory', 
                    'mode': 'd', 'value': './checkpoints'}
)
def train_i3d_widget(
    max_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    tile_size: int = 64,
    in_chans: int = 30,
    checkpoint_dir: str = './checkpoints'
):
    """Widget for training i3d model"""
    
    viewer = napari.current_viewer()
    if viewer is None:
        print("Error: No active napari viewer found.")
        return
    
    # Validate layers
    image_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Image)]
    label_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Labels)]
    
    if not image_layers:
        print("Error: No image layers found.")
        return
    
    if not label_layers:
        print("Error: No label layers found.")
        return
    
    # Check for proper pairing
    valid_pairs = False
    for img_layer in image_layers:
        for lbl_layer in label_layers:
            if lbl_layer.name.startswith(f"{img_layer.name}_"):
                valid_pairs = True
                print(f"Found pair: {img_layer.name} -> {lbl_layer.name}")
                break
    
    if not valid_pairs:
        print("Error: No valid image-label pairs found.")
        print("Label layers should be named: {image_name}_{target_name}")
        return
    
    print(f"\nStarting i3d training...")
    print(f"Epochs: {max_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Tile size: {tile_size}")
    print(f"Input channels: {in_chans}")
    
    try:
        best_checkpoint = train_i3d_model(
            viewer=viewer,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tile_size=tile_size,
            in_chans=in_chans,
            checkpoint_dir=checkpoint_dir
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Best checkpoint saved: {best_checkpoint}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point for i3d napari trainer"""
    viewer = napari.Viewer()
    
    # Set default brush size for existing and newly added Labels layers
    try:
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.brush_size = 100
        def _set_brush_size_on_add(event):
            try:
                layer = event.value
                if isinstance(layer, napari.layers.Labels):
                    layer.brush_size = 100
                    print(f"Default brush size set to 100 for '{layer.name}'")
            except Exception as e:
                print(f"Failed to set default brush size: {e}")
        # Hook when new layers are added to the viewer
        if hasattr(viewer.layers.events, 'inserted'):
            viewer.layers.events.inserted.connect(_set_brush_size_on_add)
        else:
            print("Viewer does not support 'inserted' event; dynamic brush sizing may be unavailable.")
    except Exception as e:
        print(f"Brush size initialization skipped: {e}")
    
    # Add widgets to viewer
    viewer.window.add_dock_widget(train_i3d_widget, area='right', name="i3d Training")
    viewer.window.add_dock_widget(i3d_inference_widget, area='right', name="i3d Inference")
    
    print("i3d Napari Trainer ready!")
    print("\nInstructions:")
    print("1. Load your image data as Image layers")
    print("2. Load corresponding labels as Label layers")
    print("3. Name labels as: {image_name}_{target_name}")
    print("4. Configure training parameters and click 'Train i3d Model'")
    print("5. For inference, select a model checkpoint and input layer")
    
    napari.run()

if __name__ == "__main__":
    main()