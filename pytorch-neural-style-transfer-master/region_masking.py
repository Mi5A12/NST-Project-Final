#!/usr/bin/env python3
"""
Region Masking Module for Selective Style Transfer
Allows users to apply styles to specific regions (sleeves, collars, etc.)
"""

import sys
import warnings
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple, List, Dict, Optional
import json
import os
from pathlib import Path

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

class RegionMasker:
    """Handles region masking for selective style transfer"""
    
    def __init__(self):
        self.region_types = {
            "sleeves": "Sleeves and arm areas",
            "collar": "Collar and neck area", 
            "body": "Main body/torso area",
            "hem": "Bottom hem area",
            "custom": "Custom user-defined region"
        }
        
        # Predefined region templates
        self.templates = self._create_region_templates()
    
    def _create_region_templates(self) -> Dict[str, np.ndarray]:
        """Create predefined region templates for common garment parts"""
        templates = {}
        
        # Create a base template (256x256)
        base_size = 256
        
        # Sleeves template (left and right arm areas)
        sleeves_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        # Left sleeve
        sleeves_mask[50:200, 0:60] = 255
        # Right sleeve  
        sleeves_mask[50:200, 196:256] = 255
        templates["sleeves"] = sleeves_mask
        
        # Collar template (upper center area)
        collar_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        collar_mask[0:80, 80:176] = 255
        # Add some curve to make it more collar-like
        try:
            cv2.ellipse(collar_mask, (128, 40), (48, 30), 0, 0, 180, 255, -1)
        except:
            # Fallback to simple rectangle if ellipse fails
            collar_mask[20:60, 80:176] = 255
        templates["collar"] = collar_mask
        
        # Body template (center torso area)
        body_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        body_mask[80:220, 60:196] = 255
        templates["body"] = body_mask
        
        # Hem template (bottom area)
        hem_mask = np.zeros((base_size, base_size), dtype=np.uint8)
        hem_mask[200:256, 40:216] = 255
        templates["hem"] = hem_mask
        
        return templates
    
    def create_region_mask(self, image_size: Tuple[int, int], region_type: str, 
                          custom_points: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Create a region mask for the specified area"""
        height, width = image_size
        
        if region_type == "custom" and custom_points:
            return self._create_custom_mask((height, width), custom_points)
        elif region_type in self.templates:
            # Scale template to image size
            template = self.templates[region_type]
            try:
                scaled_mask = cv2.resize(template, (width, height))
            except:
                # Fallback to simple numpy resize
                scale_y = height / template.shape[0]
                scale_x = width / template.shape[1]
                # Simple nearest neighbor resize
                new_template = np.zeros((height, width), dtype=np.uint8)
                for i in range(height):
                    for j in range(width):
                        src_i = int(i / scale_y)
                        src_j = int(j / scale_x)
                        if src_i < template.shape[0] and src_j < template.shape[1]:
                            new_template[i, j] = template[src_i, src_j]
                scaled_mask = new_template
            return scaled_mask
        else:
            # Default to full image
            return np.ones((height, width), dtype=np.uint8) * 255
    
    def _create_custom_mask(self, image_size: Tuple[int, int], 
                           points: List[Tuple[int, int]]) -> np.ndarray:
        """Create a custom mask from user-defined points"""
        height, width = image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if len(points) >= 3:
            # Create polygon from points
            pts = np.array(points, np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            # If not enough points, create a rectangle
            if len(points) == 2:
                cv2.rectangle(mask, points[0], points[1], 255, -1)
            else:
                # Default to full image
                mask.fill(255)
        
        return mask
    
    def apply_region_mask(self, content_img: torch.Tensor, style_img: torch.Tensor,
                         mask: np.ndarray, blend_strength: float = 1.0) -> torch.Tensor:
        """Apply region masking to blend content and style images"""
        # Convert mask to tensor - use torch.tensor for compatibility
        mask_array = np.asarray(mask, dtype=np.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32) / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Ensure mask is on the same device as images
        mask_tensor = mask_tensor.to(content_img.device)
        
        # Resize mask to match image dimensions
        if mask_tensor.shape[-2:] != content_img.shape[-2:]:
            mask_tensor = F.interpolate(mask_tensor, size=content_img.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        
        # Blend images based on mask
        blended = content_img * (1 - mask_tensor * blend_strength) + \
                 style_img * (mask_tensor * blend_strength)
        
        return blended
    
    def create_interactive_mask(self, image: np.ndarray, region_type: str = "custom") -> np.ndarray:
        """Create an interactive mask using OpenCV (for future GUI integration)"""
        height, width = image.shape[:2]
        
        if region_type == "custom":
            # For now, return a simple center rectangle
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = width // 2, height // 2
            cv2.rectangle(mask, 
                         (center_x - 50, center_y - 50), 
                         (center_x + 50, center_y + 50), 
                         255, -1)
            return mask
        else:
            return self.create_region_mask((height, width), region_type)

class StyleIntensityController:
    """Controls style intensity and blending"""
    
    def __init__(self):
        self.intensity_levels = {
            "subtle": 0.3,
            "moderate": 0.6, 
            "strong": 0.8,
            "maximum": 1.0
        }
    
    def adjust_style_intensity(self, style_features: List[torch.Tensor], 
                             intensity: float) -> List[torch.Tensor]:
        """Adjust style features based on intensity level"""
        adjusted_features = []
        
        for feature in style_features:
            # Scale style features by intensity
            adjusted = feature * intensity
            adjusted_features.append(adjusted)
        
        return adjusted_features
    
    def blend_styles(self, content_img: torch.Tensor, style_img: torch.Tensor,
                    intensity: float) -> torch.Tensor:
        """Blend content and style images based on intensity"""
        # Simple linear blending
        blended = content_img * (1 - intensity) + style_img * intensity
        return blended

class CreativeControls:
    """Main creative controls for advanced style transfer"""
    
    def __init__(self):
        self.region_masker = RegionMasker()
        self.intensity_controller = StyleIntensityController()
        
        # Default creative settings
        self.default_settings = {
            "style_intensity": 0.7,
            "content_weight": 1.0,
            "style_weight": 1.0,
            "tv_weight": 1.0,
            "region_masking": False,
            "selected_regions": ["body"],
            "blend_strength": 0.8
        }
    
    def apply_creative_controls(self, content_img: torch.Tensor, style_img: torch.Tensor,
                              settings: Dict) -> torch.Tensor:
        """Apply all creative controls to the style transfer"""
        result = content_img.clone()
        
        # Apply style intensity
        if "style_intensity" in settings:
            intensity = settings["style_intensity"]
            result = self.intensity_controller.blend_styles(content_img, style_img, intensity)
        
        # Apply region masking if enabled
        if settings.get("region_masking", False):
            selected_regions = settings.get("selected_regions", ["body"])
            
            for region in selected_regions:
                # Create mask for this region
                mask = self.region_masker.create_region_mask(
                    (content_img.shape[-2], content_img.shape[-1]), 
                    region
                )
                
                # Apply region-specific blending
                blend_strength = settings.get("blend_strength", 0.8)
                result = self.region_masker.apply_region_mask(
                    result, style_img, mask, blend_strength
                )
        
        return result
    
    def get_creative_presets(self) -> Dict[str, Dict]:
        """Get predefined creative presets"""
        return {
            "subtle_elegance": {
                "style_intensity": 0.4,
                "content_weight": 1.2,
                "style_weight": 0.8,
                "tv_weight": 1.5,
                "region_masking": True,
                "selected_regions": ["body"],
                "blend_strength": 0.6
            },
            "artistic_bold": {
                "style_intensity": 0.9,
                "content_weight": 0.8,
                "style_weight": 1.5,
                "tv_weight": 0.8,
                "region_masking": False,
                "blend_strength": 1.0
            },
            "selective_styling": {
                "style_intensity": 0.7,
                "content_weight": 1.0,
                "style_weight": 1.0,
                "tv_weight": 1.0,
                "region_masking": True,
                "selected_regions": ["sleeves", "collar"],
                "blend_strength": 0.8
            },
            "texture_focus": {
                "style_intensity": 0.6,
                "content_weight": 1.0,
                "style_weight": 1.2,
                "tv_weight": 0.5,
                "region_masking": True,
                "selected_regions": ["body", "hem"],
                "blend_strength": 0.7
            }
        }
    
    def save_creative_settings(self, settings: Dict, filename: str):
        """Save creative settings to file"""
        settings_path = Path("creative_settings")
        settings_path.mkdir(exist_ok=True)
        
        filepath = settings_path / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(settings, f, indent=2)
    
    def load_creative_settings(self, filename: str) -> Dict:
        """Load creative settings from file"""
        settings_path = Path("creative_settings")
        filepath = settings_path / f"{filename}.json"
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            return self.default_settings.copy()

def demo_creative_controls():
    """Demo the creative controls functionality"""
    print("🎨 Creative Controls Demo")
    print("=" * 40)
    
    # Initialize creative controls
    creative = CreativeControls()
    
    # Show available presets
    presets = creative.get_creative_presets()
    print(f"Available presets: {list(presets.keys())}")
    
    # Show region types
    region_types = creative.region_masker.region_types
    print(f"Available regions: {list(region_types.keys())}")
    
    # Test region masking
    test_mask = creative.region_masker.create_region_mask((256, 256), "sleeves")
    print(f"Sleeves mask shape: {test_mask.shape}, unique values: {np.unique(test_mask)}")
    
    print("✅ Creative controls demo completed!")

if __name__ == "__main__":
    demo_creative_controls()
