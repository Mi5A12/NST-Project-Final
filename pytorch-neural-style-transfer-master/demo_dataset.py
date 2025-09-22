#!/usr/bin/env python3
"""
Demo script for Dataset & Preprocessing features
"""

import sys
import warnings
from dataset_preprocessing import DatasetManager, PreprocessingPipeline, BackgroundRemover, DataAugmentation
from u2net_model import U2NETPredictor
import cv2
import numpy as np
from PIL import Image

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

def demo_dataset_management():
    """Demo dataset management functionality"""
    print("📊 Dataset Management Demo")
    print("=" * 40)
    
    # Initialize dataset manager
    dm = DatasetManager()
    
    # Show current stats
    stats = dm.get_dataset_stats()
    print(f"Current dataset stats: {stats}")
    
    # List available images
    garment_images = dm.list_images("garments")
    style_images = dm.list_images("styles")
    
    print(f"Found {len(garment_images)} garment images")
    print(f"Found {len(style_images)} style images")
    
    return dm

def demo_background_removal():
    """Demo background removal functionality"""
    print("\n🖼️ Background Removal Demo")
    print("=" * 40)
    
    # Initialize background remover
    br = BackgroundRemover()
    
    # Create a sample image (white background with colored object)
    sample_image = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
    cv2.circle(sample_image, (100, 100), 50, (255, 0, 0), -1)  # Blue circle
    
    print("Sample image created (white background with blue circle)")
    
    # Test threshold-based removal
    result_threshold, mask_threshold = br.remove_background(sample_image, method="threshold")
    print(f"Threshold method: Result shape {result_threshold.shape}, Mask shape {mask_threshold.shape}")
    
    # Test GrabCut removal
    result_grabcut, mask_grabcut = br.remove_background(sample_image, method="grabcut")
    print(f"GrabCut method: Result shape {result_grabcut.shape}, Mask shape {mask_grabcut.shape}")
    
    return br

def demo_data_augmentation():
    """Demo data augmentation functionality"""
    print("\n🔄 Data Augmentation Demo")
    print("=" * 40)
    
    # Initialize augmentation
    da = DataAugmentation()
    
    # Create a sample image
    sample_image = Image.new('RGB', (100, 100), color='red')
    
    print("Original image: 100x100 red square")
    
    # Test different augmentations
    augmentations = ["flip_horizontal", "color_jitter", "rotation"]
    
    for aug in augmentations:
        augmented = da.augment_image(sample_image, [aug])
        print(f"{aug}: {augmented.size}")
    
    return da

def demo_preprocessing_pipeline():
    """Demo preprocessing pipeline"""
    print("\n⚙️ Preprocessing Pipeline Demo")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    
    # Show preprocessing summary
    summary = pipeline.get_preprocessing_summary()
    print(f"Preprocessing summary: {summary}")
    
    return pipeline

def demo_u2net_model():
    """Demo U²-Net model"""
    print("\n🧠 U²-Net Model Demo")
    print("=" * 40)
    
    try:
        # Initialize U²-Net predictor
        predictor = U2NETPredictor()
        
        # Create sample image
        sample_image = np.ones((320, 320, 3), dtype=np.uint8) * 128  # Gray background
        cv2.circle(sample_image, (160, 160), 80, (255, 0, 0), -1)  # Blue circle
        
        print("Sample image created for U²-Net testing")
        
        # Test background removal
        result, mask = predictor.remove_background(sample_image)
        print(f"U²-Net result: {result.shape}, mask: {mask.shape}")
        
        return predictor
        
    except Exception as e:
        print(f"U²-Net demo failed: {e}")
        return None

def main():
    """Run all demos"""
    print("🎨 Neural Style Transfer - Dataset & Preprocessing Demo")
    print("=" * 60)
    
    try:
        # Demo dataset management
        dm = demo_dataset_management()
        
        # Demo background removal
        br = demo_background_removal()
        
        # Demo data augmentation
        da = demo_data_augmentation()
        
        # Demo preprocessing pipeline
        pipeline = demo_preprocessing_pipeline()
        
        # Demo U²-Net model
        u2net = demo_u2net_model()
        
        print("\n✅ All demos completed successfully!")
        print("\n🚀 Ready to use the full dataset and preprocessing pipeline!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
