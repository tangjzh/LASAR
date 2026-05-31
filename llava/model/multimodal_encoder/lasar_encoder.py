# Copyright 2024 LASAR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig, SiglipImageProcessor, SiglipVisionModel

from ..vggt.models.vggt import VGGT
from .vision_encoder import VisionTower, VisionTowerS2


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention layer to fuse visual semantic features and geometric features.
    F'_vis = F_vis + CrossAttn(F_vis, F_geo, F_geo)
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vis_features, geo_features):
        """
        Args:
            vis_features: Visual semantic features from Siglip, shape [B, N_vis, D]
            geo_features: Geometric features from VGGT, shape [B, N_geo, D]
        
        Returns:
            Fused features with residual connection, shape [B, N_vis, D]
        """
        # Cross-attention: Query from vis, Key and Value from geo
        attn_output, _ = self.cross_attention(
            query=vis_features,
            key=geo_features,
            value=geo_features
        )
        
        # Residual connection
        output = vis_features + attn_output
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class LASARVisionTower(VisionTower):
    """
    LASAR Vision Tower that fuses:
    1. Semantic features from Siglip encoder
    2. Geometric features from VGGT encoder
    Using cross-attention mechanism.
    """
    def __init__(self, siglip_model_path: str, config: PretrainedConfig, vggt_model_path: str = "facebook/VGGT-1B", 
                 _skip_load: bool = False):
        super().__init__(siglip_model_path, config)
        
        if not _skip_load:
            # Load Siglip vision encoder for semantic features
            self.image_processor = SiglipImageProcessor.from_pretrained(siglip_model_path)
            self.siglip_tower = SiglipVisionModel.from_pretrained(
                siglip_model_path, 
                torch_dtype=eval(config.model_dtype)
            )
            
            # Load VGGT encoder for geometric features
            print(f"Loading VGGT model from {vggt_model_path}...")
            self.vggt_tower = VGGT.from_pretrained(vggt_model_path)
            
            # Freeze VGGT encoder (it's pre-trained on point cloud prediction)
            # for param in self.vggt_tower.parameters():
            #     param.requires_grad = False
            # self.vggt_tower.eval()
            
            # Get hidden dimensions
            self.siglip_hidden_size = self.siglip_tower.config.hidden_size
            
            # VGGT typically outputs 2048-dim features, but we need to match it with Siglip
            # Add a projection layer to align VGGT features to Siglip dimension
            self.vggt_proj = nn.Linear(2048, self.siglip_hidden_size)
            
            # Cross-attention fusion module
            self.fusion_module = CrossAttentionFusion(
                hidden_dim=self.siglip_hidden_size,
                num_heads=8,
                dropout=0.0
            )
            
            self.is_loaded = True
            
            # Override vision_tower to use siglip for compatibility
            self.vision_tower = self.siglip_tower
        else:
            # Minimal init for from_pretrained
            self.is_loaded = False
    
    def encode_siglip(self, images):
        """Extract semantic features using Siglip encoder."""
        outputs = self.siglip_tower(
            pixel_values=images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        # Get features from the specified layer
        vis_features = self.feature_select(outputs)
        return vis_features
    
    def encode_vggt(self, images):
        """
        Extract geometric features using VGGT encoder.
        Note: VGGT expects images in shape [B, 3, 224, 224]
        """
        with torch.no_grad():
            # VGGT is frozen, no gradient needed
            # Resize images to 224x224 if needed (VGGT's expected input size)
            if images.shape[-1] != 224:
                images_224 = torch.nn.functional.interpolate(
                    images, 
                    size=(224, 224), 
                    mode='bicubic', 
                    align_corners=False
                )
            else:
                images_224 = images
            
            # VGGT forward pass
            geo_features, patch_start_idx = self.vggt_tower(images_224.to(device=self.device, dtype=self.dtype))
            geo_features = geo_features[:, :, patch_start_idx:]

        # Project VGGT features to match Siglip dimension
        geo_features = self.vggt_proj(geo_features.to(self.dtype)).squeeze(0)
        
        return geo_features
    
    def forward(self, images):
        """
        Forward pass that fuses semantic and geometric features.
        
        Args:
            images: Input images, can be a list or a tensor
        
        Returns:
            Fused visual features with geometry awareness
        """
        if type(images) is list:
            fused_features = []
            for image in images:
                # Extract semantic features
                vis_features = self.encode_siglip(image.unsqueeze(0))
                
                # Extract geometric features  
                geo_features = self.encode_vggt(image.unsqueeze(0))
                
                # Fuse using cross-attention
                fused = self.fusion_module(vis_features, geo_features)
                fused_features.append(fused)
            
            return fused_features
        else:
            # Batch processing
            # Extract semantic features from Siglip
            vis_features = self.encode_siglip(images)
            
            # Extract geometric features from VGGT
            geo_features = self.encode_vggt(images)

            # Fuse features using cross-attention: F'_vis = F_vis + CrossAttn(F_vis, F_geo, F_geo)
            fused_features = self.fusion_module(vis_features, geo_features)
            
            return fused_features
    
    @property
    def hidden_size(self):
        """Return the hidden size of the fused features."""
        return self.siglip_hidden_size
    
    @property
    def config(self):
        """Return the config of the Siglip tower for compatibility."""
        if self.is_loaded:
            return self.siglip_tower.config
        else:
            return self.cfg_only
    
    def save_pretrained(self, output_dir, state_dict=None):
        """
        Save the complete LASAR vision tower including:
        1. Siglip encoder
        2. VGGT encoder  
        3. VGGT projection layer
        4. Fusion module
        """
        from collections import OrderedDict
        
        os.makedirs(output_dir, exist_ok=True)
        
        # If state_dict is provided, extract component-specific state dicts
        if state_dict is not None:
            # Extract Siglip state dict
            siglip_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('siglip_tower.'):
                    new_key = k.replace('siglip_tower.', '')
                    siglip_state_dict[new_key] = v
            
            # Extract VGGT state dict
            vggt_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('vggt_tower.'):
                    new_key = k.replace('vggt_tower.', '')
                    vggt_state_dict[new_key] = v
            
            # Extract trainable components
            vggt_proj_state_dict = OrderedDict()
            fusion_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('vggt_proj.'):
                    new_key = k.replace('vggt_proj.', '')
                    vggt_proj_state_dict[new_key] = v
                elif k.startswith('fusion_module.'):
                    new_key = k.replace('fusion_module.', '')
                    fusion_state_dict[new_key] = v
        else:
            siglip_state_dict = None
            vggt_state_dict = None
            vggt_proj_state_dict = self.vggt_proj.state_dict()
            fusion_state_dict = self.fusion_module.state_dict()
        
        # Save Siglip encoder
        siglip_dir = os.path.join(output_dir, "siglip")
        os.makedirs(siglip_dir, exist_ok=True)
        if siglip_state_dict:
            self.siglip_tower.save_pretrained(siglip_dir, state_dict=siglip_state_dict)
        else:
            self.siglip_tower.save_pretrained(siglip_dir)
        self.image_processor.save_pretrained(siglip_dir)
        print(f"✓ Saved Siglip encoder to {siglip_dir}")
        
        # Save VGGT encoder
        # Note: VGGT has shared tensors that cause safetensors issues, use pytorch format
        vggt_dir = os.path.join(output_dir, "vggt")
        os.makedirs(vggt_dir, exist_ok=True)
        
        # Always use manual save for VGGT to avoid safetensors issues
        if vggt_state_dict:
            torch.save(vggt_state_dict, os.path.join(vggt_dir, "pytorch_model.bin"))
        else:
            torch.save(self.vggt_tower.state_dict(), os.path.join(vggt_dir, "pytorch_model.bin"))
        # Save config if available
        if hasattr(self.vggt_tower, 'config'):
            self.vggt_tower.config.save_pretrained(vggt_dir)
        print(f"✓ Saved VGGT encoder to {vggt_dir} (manual)")
        
        # Save trainable components (vggt_proj and fusion_module)
        trainable_state = {
            'vggt_proj': vggt_proj_state_dict,
            'fusion_module': fusion_state_dict,
        }
        torch.save(trainable_state, os.path.join(output_dir, "lasar_trainable.pt"))
        print(f"✓ Saved LASAR trainable components to {output_dir}/lasar_trainable.pt")
        
        # Save config info
        lasar_config = {
            'vision_tower_type': 'LASARVisionTower',
            'siglip_path': 'siglip',  # relative path
            'vggt_path': 'vggt',  # relative path
            'siglip_hidden_size': self.siglip_hidden_size,
        }
        torch.save(lasar_config, os.path.join(output_dir, "lasar_config.pt"))
        print(f"✓ Saved LASAR config to {output_dir}/lasar_config.pt")
    
    @classmethod
    def from_pretrained(cls, pretrained_path, config):
        """
        Load the complete LASAR vision tower from checkpoint.
        """
        lasar_config_path = os.path.join(pretrained_path, "lasar_config.pt")
        if not os.path.exists(lasar_config_path):
            lasar_config_path = os.path.join(pretrained_path, "veme_config.pt")
        if os.path.exists(lasar_config_path):
            lasar_config = torch.load(lasar_config_path, map_location='cpu')
            print(f"✓ Loading LASAR vision tower from {pretrained_path}")
            
            # Load from saved checkpoint
            siglip_path = os.path.join(pretrained_path, lasar_config['siglip_path'])
            vggt_path = os.path.join(pretrained_path, lasar_config['vggt_path'])
            
            # Load Siglip
            from transformers import SiglipImageProcessor, SiglipVisionModel
            image_processor = SiglipImageProcessor.from_pretrained(siglip_path)
            siglip_tower = SiglipVisionModel.from_pretrained(
                siglip_path, 
                torch_dtype=eval(config.model_dtype)
            )
            
            # Load VGGT - handle both save_pretrained and manual save
            from ..vggt.models.vggt import VGGT
            vggt_manual_path = os.path.join(vggt_path, "pytorch_model.bin")
            if os.path.exists(vggt_manual_path):
                # Manual save format
                print(f"✓ Loading VGGT from manual save")
                vggt_tower = VGGT.from_pretrained("facebook/VGGT-1B")
                vggt_state = torch.load(vggt_manual_path, map_location='cpu')
                vggt_tower.load_state_dict(vggt_state)
            else:
                # Standard save_pretrained format
                print(f"✓ Loading VGGT from save_pretrained")
                vggt_tower = VGGT.from_pretrained(vggt_path)
            
            # Create instance with skip_load=True to avoid double loading
            instance = cls(siglip_path, config, vggt_path, _skip_load=True)
            
            # Set loaded components
            instance.image_processor = image_processor
            instance.siglip_tower = siglip_tower
            instance.vggt_tower = vggt_tower
            instance.siglip_hidden_size = siglip_tower.config.hidden_size
            
            # Initialize projections and fusion
            import torch.nn as nn
            instance.vggt_proj = nn.Linear(2048, instance.siglip_hidden_size)
            instance.fusion_module = CrossAttentionFusion(
                hidden_dim=instance.siglip_hidden_size,
                num_heads=8,
                dropout=0.0
            )
            
            instance.is_loaded = True
            instance.vision_tower = instance.siglip_tower
            
            # Load trainable components
            trainable_path = os.path.join(pretrained_path, "lasar_trainable.pt")
            if os.path.exists(trainable_path):
                trainable_state = torch.load(trainable_path, map_location='cpu')
                instance.vggt_proj.load_state_dict(trainable_state['vggt_proj'])
                instance.fusion_module.load_state_dict(trainable_state['fusion_module'])
                print(f"✓ Loaded LASAR trainable components")
            
            return instance
        else:
            # Fallback: create new instance with default paths
            print(f"⚠️  No LASAR config found, creating new LASAR tower")
            return cls(
                siglip_model_path="google/siglip-so400m-patch14-384",
                config=config,
                vggt_model_path="facebook/VGGT-1B"
            )


class LASARVisionTowerS2(VisionTowerS2):
    """
    LASAR Vision Tower with multi-scale support (S2).
    Extends VisionTowerS2 to support the LASAR dual-stream architecture.
    """
    def __init__(self, siglip_model_path: str, config: PretrainedConfig, vggt_model_path: str = "facebook/VGGT-1B"):
        super().__init__(siglip_model_path, config)
        
        # Load Siglip vision encoder
        self.image_processor = AutoImageProcessor.from_pretrained(siglip_model_path)
        self.siglip_tower = AutoModel.from_pretrained(
            siglip_model_path, 
            torch_dtype=eval(config.model_dtype)
        )
        
        # Adjust image processor for multi-scale
        if hasattr(self.image_processor, 'size'):
            self.image_processor.size['height'] = self.scales[-1]
            self.image_processor.size['width'] = self.scales[-1]
        
        # Load VGGT encoder
        print(f"Loading VGGT model from {vggt_model_path}...")
        self.vggt_tower = VGGT.from_pretrained(vggt_model_path)
        
        # Freeze VGGT
        for param in self.vggt_tower.parameters():
            param.requires_grad = False
        self.vggt_tower.eval()
        
        # Get dimensions
        self.siglip_hidden_size = self.siglip_tower.config.hidden_size
        
        # Projection and fusion layers
        self.vggt_proj = nn.Linear(2048, self.siglip_hidden_size)
        self.fusion_module = CrossAttentionFusion(
            hidden_dim=self.siglip_hidden_size,
            num_heads=8,
            dropout=0.0
        )
        
        self.is_loaded = True
        self.vision_tower = self.siglip_tower
    
    @torch.no_grad()
    def forward_feature(self, images):
        """
        Forward pass for multi-scale processing.
        This will be called by multiscale_forward wrapper.
        """
        # Extract semantic features
        siglip_outputs = self.siglip_tower(
            pixel_values=images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        vis_features = self.feature_select(siglip_outputs).to(images.dtype)
        
        # Extract geometric features
        if images.shape[-1] != 224:
            images_224 = torch.nn.functional.interpolate(
                images, size=(224, 224), mode='bicubic', align_corners=False
            )
        else:
            images_224 = images
        
        geo_features = self.vggt_tower(images_224.to(device=self.device, dtype=self.dtype))
        if isinstance(geo_features, dict):
            geo_features = geo_features.get('features', geo_features.get('last_hidden_state'))
        
        geo_features = self.vggt_proj(geo_features.to(self.dtype)).squeeze(0)
        
        # Fuse features
        fused_features = self.fusion_module(vis_features, geo_features)
        
        return fused_features
    
    @property
    def hidden_size(self):
        """For S2, return the total hidden size across all scales."""
        return self.siglip_hidden_size * len(self.scales)

