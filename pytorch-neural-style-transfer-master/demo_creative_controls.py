#!/usr/bin/env python3
"""
Demo script for Creative Controls features
"""

import sys
import warnings
import numpy as np
import torch
from region_masking import CreativeControls, RegionMasker, StyleIntensityController
from region_preview import RegionPreview
import cv2

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

def demo_style_intensity():
    """Demo style intensity controls"""
    print("🎨 Style Intensity Demo")
    print("=" * 40)
    
    # Create sample tensors
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)
    
    # Test different intensity levels
    controller = StyleIntensityController()
    
    for intensity in [0.3, 0.6, 0.9]:
        blended = controller.blend_styles(content_img, style_img, intensity)
        print(f"Intensity {intensity}: Blended tensor shape {blended.shape}")
    
    print("✅ Style intensity demo completed!")

def demo_region_masking():
    """Demo region masking functionality"""
    print("\n🎯 Region Masking Demo")
    print("=" * 40)
    
    # Create sample image
    sample_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # Test region masker
    masker = RegionMasker()
    
    # Test different regions
    regions = ["sleeves", "collar", "body", "hem"]
    
    for region in regions:
        mask = masker.create_region_mask((256, 256), region)
        unique_values = np.unique(mask)
        print(f"{region}: Mask shape {mask.shape}, unique values {unique_values}")
    
    print("✅ Region masking demo completed!")

def demo_region_preview():
    """Demo region preview functionality"""
    print("\n👁️ Region Preview Demo")
    print("=" * 40)
    
    # Create sample garment image
    sample_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # Add garment features using numpy
    sample_image[80:220, 60:196] = [200, 200, 200]  # Body
    sample_image[50:200, 0:60] = [180, 180, 180]     # Left sleeve
    sample_image[50:200, 196:256] = [180, 180, 180]  # Right sleeve
    sample_image[0:80, 80:176] = [220, 220, 220]     # Collar
    
    # Test region preview
    preview = RegionPreview()
    test_regions = ["sleeves", "collar", "body"]
    
    result = preview.create_region_preview(sample_image, test_regions)
    print(f"Preview created with regions: {test_regions}")
    print(f"Preview shape: {result.shape}")
    
    legend = preview.create_region_legend(test_regions)
    print(f"Legend:\n{legend}")
    
    print("✅ Region preview demo completed!")

def demo_creative_presets():
    """Demo creative presets"""
    print("\n🎭 Creative Presets Demo")
    print("=" * 40)
    
    creative = CreativeControls()
    presets = creative.get_creative_presets()
    
    print("Available presets:")
    for name, settings in presets.items():
        print(f"\n{name}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("✅ Creative presets demo completed!")

def demo_creative_controls():
    """Demo the complete creative controls system"""
    print("\n🎨 Complete Creative Controls Demo")
    print("=" * 40)
    
    # Initialize creative controls
    creative = CreativeControls()
    
    # Create sample images
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)
    
    # Test different creative settings
    test_settings = [
        {
            "style_intensity": 0.5,
            "content_weight": 1.0,
            "style_weight": 1.0,
            "tv_weight": 1.0,
            "region_masking": False
        },
        {
            "style_intensity": 0.8,
            "content_weight": 0.8,
            "style_weight": 1.2,
            "tv_weight": 0.8,
            "region_masking": True,
            "selected_regions": ["sleeves", "collar"],
            "blend_strength": 0.7
        }
    ]
    
    for i, settings in enumerate(test_settings):
        print(f"\nTest {i+1}: {settings}")
        result = creative.apply_creative_controls(content_img, style_img, settings)
        print(f"Result shape: {result.shape}")
    
    print("✅ Complete creative controls demo completed!")

def main():
    """Run all creative controls demos"""
    print("🎨 Neural Style Transfer - Creative Controls Demo")
    print("=" * 60)
    
    try:
        # Run individual demos
        demo_style_intensity()
        demo_region_masking()
        demo_region_preview()
        demo_creative_presets()
        demo_creative_controls()
        
        print("\n🎉 All creative controls demos completed successfully!")
        print("\n🚀 Ready to use the full creative controls system!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
