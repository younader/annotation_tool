import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import napari
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Dict, Any, Union
import cv2
import warnings
import sys

class Decoder(nn.Module):
    """Decoder from the i3d implementation"""
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=False)

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear", align_corners=False)
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class I3DSegmentationModel(nn.Module):
    """Combined i3d encoder + decoder model"""
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        
    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]  # Add channel dimension
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask


class I3DModelLoader:
    """
    Loader for InceptionI3d models from checkpoints.
    """
    
    def __init__(self, checkpoint_path, enc: str = 'i3d'):
        self.checkpoint_path = Path(checkpoint_path)
        self.encoder_type = enc
        
    def load(self):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Try multiple import strategies based on encoder type
        if self.encoder_type == 'i3d':
            try:
                from models.i3dallnl import InceptionI3d
                print("Successfully imported InceptionI3d from models.i3dallnl")
            except ImportError:
                try:
                    from .training import I3DLightningModel
                    print("Using I3DLightningModel from training module")
                    model = I3DLightningModel.load_from_checkpoint(
                        self.checkpoint_path,
                        learning_rate=2e-5,
                        in_channels=1,
                        size=64,
                        enc='i3d'
                    )
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    model.eval()
                    print(f"Model loaded and moved to {device}")
                    return model
                except Exception as e:
                    print(f"Could not load from training module: {e}")
                    print("Creating InceptionI3d model from scratch")
                    from .i3d_standalone import InceptionI3d
        else:
            # resnet3d path
            try:
                from models.resnetall import generate_model
                print("Successfully imported resnet3d generate_model")
            except ImportError as e:
                print(f"Failed to import resnet3d: {e}")
                raise
        
        # Create model instance
        if self.encoder_type == 'i3d':
            print("Creating InceptionI3d model instance...")
            model = InceptionI3d(in_channels=1, num_classes=512, non_local=True)
        else:
            print("Creating resnet3d model instance...")
            model = generate_model(model_depth=50, n_input_channels=1, forward_features=True, n_classes=1039)
        
        # Build decoder
        print("Building decoder...")
        with torch.no_grad():
            test_input = torch.rand(1, 1, 20, 256, 256)
            encoder_dims = [x.size(1) for x in model(test_input)]
        decoder = Decoder(encoder_dims=encoder_dims, upscale=1)
        
        # Combine into full model
        full_model = I3DSegmentationModel(model, decoder)
        
        # Load checkpoint
        print("Loading checkpoint weights...")
        checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
            print("Found state_dict in checkpoint")
        elif 'model_state_dict' in checkpoint_data:
            state_dict = checkpoint_data['model_state_dict']
            print("Found model_state_dict in checkpoint")
        else:
            state_dict = checkpoint_data
            print("Using checkpoint as state_dict directly")
        
        # Try to load weights
        try:
            full_model.load_state_dict(state_dict, strict=False)
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load full state dict: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded weights into backbone only")
            except Exception as e2:
                print(f"Could not load weights, using random initialization: {e2}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        full_model = full_model.to(device)
        full_model.eval()
        
        print(f"Model ready on {device}")
        return full_model


class I3DInferer:
    """
    Inference engine for i3d model using the same tiling approach as training.
    """
    def __init__(self, 
               model: nn.Module,
               tile_size: int = 64,
               stride: int = 32,
               in_chans: int = 30,
               batch_size: int = 16,
               device = None):
        
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.tile_size = tile_size
        self.stride = stride
        self.in_chans = in_chans
        self.batch_size = batch_size
        
        print(f"Inferer initialized - tile_size: {tile_size}, stride: {stride}, device: {self.device}")
        
    def infer(self, image_data: np.ndarray) -> np.ndarray:
        """
        Run inference on image data using i3d tiling approach.
        """
        
        print(f"Starting inference on image with shape: {image_data.shape}")
        
        # Handle different input formats; if uint16, rescale to approx uint8 range
        if image_data.dtype == np.uint16:
            image_data = (image_data.astype(np.float32) / 257.0).astype(np.float32)
        else:
            image_data = image_data.astype(np.float32)

        if image_data.ndim == 2:
            # Single 2D slice - replicate to create volume
            print("Input is 2D, replicating to create volume")
            image_data = np.stack([image_data] * self.in_chans, axis=2)
        elif image_data.ndim == 3:
            # Could be (D, H, W) or (H, W, D)
            # Check which dimension is likely the channel/depth
            if image_data.shape[0] < 100 and image_data.shape[1] > 100 and image_data.shape[2] > 100:
                # Likely (C, H, W) - transpose to (H, W, C)
                print(f"Transposing from (C, H, W) to (H, W, C)")
                image_data = np.transpose(image_data, (1, 2, 0))
            
            # Now handle channel dimension
            if image_data.shape[2] < self.in_chans:
                # Pad with edge values if not enough channels
                padding = self.in_chans - image_data.shape[2]
                print(f"Padding {padding} channels")
                image_data = np.pad(image_data, ((0, 0), (0, 0), (0, padding)), mode='edge')
            elif image_data.shape[2] > self.in_chans:
                # Extract subset of channels
                mid = image_data.shape[2] // 2
                start = max(0, mid - self.in_chans // 2)
                end = min(image_data.shape[2], start + self.in_chans)
                print(f"Extracting channels {start} to {end}")
                image_data = image_data[:, :, start:end]
        
        print(f"Preprocessed image shape: {image_data.shape}")
        
        # Pad image to be divisible by tile_size
        pad0 = (self.tile_size - image_data.shape[0] % self.tile_size) % self.tile_size
        pad1 = (self.tile_size - image_data.shape[1] % self.tile_size) % self.tile_size
        image_padded = np.pad(image_data, ((0, pad0), (0, pad1), (0, 0)), constant_values=0)
        
        # Clip values to uint8 dynamic range after potential rescale
        image_padded = np.clip(image_padded, 0, 255).astype(np.float32)
        
        print(f"Padded image shape: {image_padded.shape}")
        
        # Initialize prediction mask
        mask_pred = np.zeros((image_padded.shape[0], image_padded.shape[1]), dtype=np.float32)
        mask_count = np.zeros((image_padded.shape[0], image_padded.shape[1]), dtype=np.float32)
        
        # Generate tiles
        x1_list = list(range(0, image_padded.shape[1] - self.tile_size + 1, self.stride))
        y1_list = list(range(0, image_padded.shape[0] - self.tile_size + 1, self.stride))
        
        # Add final tiles if stride doesn't cover the whole image
        if len(x1_list) == 0 or x1_list[-1] + self.tile_size < image_padded.shape[1]:
            x1_list.append(max(0, image_padded.shape[1] - self.tile_size))
        if len(y1_list) == 0 or y1_list[-1] + self.tile_size < image_padded.shape[0]:
            y1_list.append(max(0, image_padded.shape[0] - self.tile_size))
        
        # Collect tiles
        tiles = []
        coords = []
        
        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + self.tile_size
                x2 = x1 + self.tile_size
                
                tile = image_padded[y1:y2, x1:x2]
                tiles.append(tile)
                coords.append((x1, y1, x2, y2))
        
        print(f"Processing {len(tiles)} tiles...")

        # Debug export removed after validation phase
        
        # Batch process tiles with progress bar
        batch_size = self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(tiles), batch_size), desc="Processing tiles"):
                batch_tiles = tiles[i:i+batch_size]
                batch_coords = coords[i:i+batch_size]
                
                # Prepare batch
                batch_tensors = []
                for tile in batch_tiles:
                    tile_tensor = self._prepare_tile(tile)
                    batch_tensors.append(tile_tensor)
                
                if len(batch_tensors) == 0:
                    continue
                    
                batch_tensor = torch.cat(batch_tensors, dim=0)
                batch_tensor = batch_tensor.to(self.device)
                
                # Run inference
                try:
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast(enabled=True):
                            outputs = self.model(batch_tensor)
                    else:
                        outputs = self.model(batch_tensor)
                    
                    outputs = torch.sigmoid(outputs)
                    
                except Exception as e:
                    print(f"Error during model forward pass: {e}")
                    continue
                
                # Process outputs
                outputs = outputs.cpu()
                
                for j, (x1, y1, x2, y2) in enumerate(batch_coords):
                    try:
                        # Upsample prediction to original size
                        pred = outputs[j:j+1]
                        pred = F.interpolate(
                            pred,
                            size=(self.tile_size, self.tile_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        pred = pred.squeeze(0).squeeze(0).numpy()
                        
                        # Accumulate predictions
                        mask_pred[y1:y2, x1:x2] += pred
                        mask_count[y1:y2, x1:x2] += 1
                    except Exception as e:
                        print(f"Error processing tile output: {e}")
                        continue
        
        print("Averaging predictions...")
        
        # Average predictions
        mask_pred = np.divide(mask_pred, mask_count, 
                             out=np.zeros_like(mask_pred), 
                             where=mask_count > 0)
        
        # Remove padding
        if pad0 > 0:
            mask_pred = mask_pred[:-pad0, :]
        if pad1 > 0:
            mask_pred = mask_pred[:, :-pad1]
        
        print(f"Inference complete. Output shape: {mask_pred.shape}")
        
        return mask_pred
    
    def _prepare_tile(self, tile):
        """Prepare a tile for model input"""
        # Ensure tile has correct shape
        if tile.shape != (self.tile_size, self.tile_size, self.in_chans):
            print(f"Warning: tile shape {tile.shape} doesn't match expected {(self.tile_size, self.tile_size, self.in_chans)}")
            return None
        
        # Normalize to match training: divide by 255 only (mean=0, std=1 in training)
        tile = tile.astype(np.float32) / 255.0
        
        # Convert to tensor (C, D, H, W) for 3D conv
        tile_tensor = torch.from_numpy(tile).float()
        tile_tensor = tile_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        tile_tensor = tile_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims -> (1, 1, C, H, W)
        
        return tile_tensor


def run_i3d_inference(viewer: napari.Viewer, 
                    layer: napari.layers.Layer, 
                    checkpoint_path: str,
                    model_type: str = 'resnet3d',
                    tile_size: int = 64,
                    stride: Optional[int] = None,
                    in_chans: int = 24,
                    batch_size: int = 16) -> List[napari.layers.Layer]:
    """
    Run i3d model inference on a napari layer.
    """
    
    print(f"\n{'='*50}")
    print(f"Starting i3d inference")
    print(f"{'='*50}")
    
    if stride is None:
        stride = 32
        print(f"Auto-calculated stride: {stride}")
    
    # Load model
    try:
        loader = I3DModelLoader(checkpoint_path, enc=model_type)
        model = loader.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Get image data
    image_data = np.array(layer.data)
    print(f"Input layer '{layer.name}' shape: {image_data.shape}, dtype: {image_data.dtype}")
    
    # Create inferer
    inferer = I3DInferer(
        model=model,
        tile_size=tile_size,
        stride=stride,
        in_chans=in_chans,
        batch_size=batch_size
    )
    
    # Run inference
    print(f"\nRunning inference...")
    try:
        prediction = inferer.infer(image_data)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    
    # Add result to viewer
    checkpoint_name = Path(checkpoint_path).stem
    layer_name = f"{layer.name}_{checkpoint_name}_prediction"
    
    print(f"\nAdding prediction to viewer as '{layer_name}'")
    
    new_layer = viewer.add_image(
        prediction,
        name=layer_name,
        colormap='viridis',
        blending='translucent',
        opacity=0.7
    )
    
    print(f"\n{'='*50}")
    print(f"Inference completed successfully!")
    print(f"Output shape: {prediction.shape}, range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"{'='*50}\n")
    
    return [new_layer]