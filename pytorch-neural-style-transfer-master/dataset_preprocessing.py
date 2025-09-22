#!/usr/bin/env python3
"""
Dataset & Preprocessing Module for Neural Style Transfer
Implements garment curation, background removal, and data augmentation
"""

import sys
import warnings
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm
import requests
from urllib.parse import urlparse

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset structure and organization"""
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = Path(base_path)
        self.garments_path = self.base_path / "garments"
        self.styles_path = self.base_path / "styles"
        self.processed_garments_path = self.base_path / "processed_garments"
        self.processed_styles_path = self.base_path / "processed_styles"
        self.masks_path = self.base_path / "masks"
        
        # Create directory structure
        self._create_directories()
        
        # Dataset statistics
        self.stats = {
            "garments": {"total": 0, "dresses": 0, "tops": 0, "skirts": 0},
            "styles": {"total": 0, "paintings": 0, "textiles": 0, "abstract": 0}
        }
    
    def _create_directories(self):
        """Create the dataset directory structure"""
        directories = [
            self.garments_path / "dresses",
            self.garments_path / "tops", 
            self.garments_path / "skirts",
            self.styles_path / "paintings",
            self.styles_path / "textiles",
            self.styles_path / "abstract",
            self.processed_garments_path,
            self.processed_styles_path,
            self.masks_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def add_garment(self, image_path: str, category: str, copy: bool = True) -> str:
        """Add a garment image to the dataset"""
        if category not in ["dresses", "tops", "skirts"]:
            raise ValueError(f"Invalid category: {category}")
        
        source_path = Path(image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate unique filename
        filename = f"{category}_{len(list((self.garments_path / category).glob('*'))):04d}{source_path.suffix}"
        dest_path = self.garments_path / category / filename
        
        if copy:
            shutil.copy2(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))
        
        logger.info(f"Added garment: {filename} to {category}")
        return str(dest_path)
    
    def add_style(self, image_path: str, category: str, copy: bool = True) -> str:
        """Add a style image to the dataset"""
        if category not in ["paintings", "textiles", "abstract"]:
            raise ValueError(f"Invalid category: {category}")
        
        source_path = Path(image_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate unique filename
        filename = f"{category}_{len(list((self.styles_path / category).glob('*'))):04d}{source_path.suffix}"
        dest_path = self.styles_path / category / filename
        
        if copy:
            shutil.copy2(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))
        
        logger.info(f"Added style: {filename} to {category}")
        return str(dest_path)
    
    def get_dataset_stats(self) -> Dict:
        """Get current dataset statistics"""
        for category in ["dresses", "tops", "skirts"]:
            count = len(list((self.garments_path / category).glob("*")))
            self.stats["garments"][category] = count
            self.stats["garments"]["total"] += count
        
        for category in ["paintings", "textiles", "abstract"]:
            count = len(list((self.styles_path / category).glob("*")))
            self.stats["styles"][category] = count
            self.stats["styles"]["total"] += count
        
        return self.stats
    
    def list_images(self, dataset_type: str, category: str = None) -> List[str]:
        """List all images in a dataset category"""
        if dataset_type == "garments":
            base_path = self.garments_path
        elif dataset_type == "styles":
            base_path = self.styles_path
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        
        if category:
            return [str(p) for p in (base_path / category).glob("*") if p.is_file()]
        else:
            all_images = []
            for cat in base_path.iterdir():
                if cat.is_dir():
                    all_images.extend([str(p) for p in cat.glob("*") if p.is_file()])
            return all_images

class BackgroundRemover:
    """Background removal using U²-Net and traditional methods"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.u2net_model = None
        self._load_u2net()
    
    def _load_u2net(self):
        """Load U²-Net model for background removal"""
        try:
            # For now, we'll use a simple threshold-based method
            # In production, you would load the actual U²-Net model
            logger.info("Using threshold-based background removal (U²-Net not available)")
            self.u2net_model = None
        except Exception as e:
            logger.warning(f"Could not load U²-Net: {e}")
            self.u2net_model = None
    
    def remove_background_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background using threshold-based method"""
        # Convert to different color spaces for better segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Create masks for different background types
        # White background
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        
        # Light background (various colors)
        light_mask = cv2.inRange(lab, (200, 0, 0), (255, 255, 255))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, light_mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Create alpha channel
        alpha = 255 - combined_mask
        
        # Apply mask to image
        result = image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = alpha
        
        return result, alpha
    
    def remove_background_grabcut(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background using GrabCut algorithm"""
        # Initialize mask
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # Define rectangle (assume object is in center)
        height, width = image.shape[:2]
        rect = (width//4, height//4, width//2, height//2)
        
        # Initialize background and foreground models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask
        result = image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = mask2 * 255
        
        return result, mask2 * 255
    
    def remove_background(self, image: np.ndarray, method: str = "threshold") -> Tuple[np.ndarray, np.ndarray]:
        """Remove background from image"""
        if method == "threshold":
            return self.remove_background_threshold(image)
        elif method == "grabcut":
            return self.remove_background_grabcut(image)
        else:
            raise ValueError(f"Unknown method: {method}")

class DataAugmentation:
    """Data augmentation for garment and style images"""
    
    def __init__(self):
        self.transform_pipeline = self._create_transform_pipeline()
    
    def _create_transform_pipeline(self):
        """Create augmentation pipeline"""
        return {
            "flip_horizontal": transforms.RandomHorizontalFlip(p=0.5),
            "flip_vertical": transforms.RandomVerticalFlip(p=0.3),
            "rotation": transforms.RandomRotation(degrees=15),
            "color_jitter": transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            "affine": transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            )
        }
    
    def augment_image(self, image: Image.Image, augmentations: List[str] = None) -> Image.Image:
        """Apply augmentations to an image"""
        if augmentations is None:
            augmentations = ["flip_horizontal", "color_jitter", "rotation"]
        
        augmented = image.copy()
        
        for aug_name in augmentations:
            if aug_name in self.transform_pipeline:
                augmented = self.transform_pipeline[aug_name](augmented)
        
        return augmented
    
    def create_augmented_dataset(self, input_dir: str, output_dir: str, 
                                num_augmentations: int = 3) -> List[str]:
        """Create augmented dataset from input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        augmented_files = []
        
        for image_file in tqdm(input_path.glob("*"), desc="Augmenting images"):
            if not image_file.is_file():
                continue
            
            try:
                # Load original image
                original = Image.open(image_file)
                
                # Save original
                original_path = output_path / f"original_{image_file.name}"
                original.save(original_path)
                augmented_files.append(str(original_path))
                
                # Create augmented versions
                for i in range(num_augmentations):
                    augmented = self.augment_image(original)
                    aug_path = output_path / f"aug_{i:02d}_{image_file.name}"
                    augmented.save(aug_path)
                    augmented_files.append(str(aug_path))
                    
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        return augmented_files

class PreprocessingPipeline:
    """Main preprocessing pipeline for neural style transfer"""
    
    def __init__(self, target_size: int = 256, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.target_size = target_size
        self.device = device
        self.dataset_manager = DatasetManager()
        self.background_remover = BackgroundRemover(device)
        self.augmentation = DataAugmentation()
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_garment(self, image_path: str, remove_bg: bool = True, 
                          augment: bool = False) -> Dict[str, str]:
        """Preprocess a garment image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove background if requested
        if remove_bg:
            image_rgba, mask = self.background_remover.remove_background(image)
            # Save mask
            mask_path = self.dataset_manager.masks_path / f"mask_{Path(image_path).stem}.png"
            cv2.imwrite(str(mask_path), mask)
        else:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            image_rgba[:, :, 3] = 255  # Full opacity
        
        # Resize to target size
        image_rgba = cv2.resize(image_rgba, (self.target_size, self.target_size))
        
        # Convert to PIL for further processing
        pil_image = Image.fromarray(image_rgba, 'RGBA')
        
        # Save processed image
        processed_path = self.dataset_manager.processed_garments_path / f"processed_{Path(image_path).name}"
        pil_image.save(processed_path)
        
        result = {
            "original": image_path,
            "processed": str(processed_path),
            "size": f"{self.target_size}x{self.target_size}"
        }
        
        if remove_bg:
            result["mask"] = str(mask_path)
        
        # Create augmented versions if requested
        if augment:
            augmented_files = self.augmentation.create_augmented_dataset(
                str(Path(image_path).parent),
                str(self.dataset_manager.processed_garments_path / "augmented"),
                num_augmentations=2
            )
            result["augmented"] = augmented_files
        
        return result
    
    def preprocess_style(self, image_path: str, augment: bool = False) -> Dict[str, str]:
        """Preprocess a style image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Save processed image
        processed_path = self.dataset_manager.processed_styles_path / f"processed_{Path(image_path).name}"
        pil_image.save(processed_path)
        
        result = {
            "original": image_path,
            "processed": str(processed_path),
            "size": f"{self.target_size}x{self.target_size}"
        }
        
        # Create augmented versions if requested
        if augment:
            augmented_files = self.augmentation.create_augmented_dataset(
                str(Path(image_path).parent),
                str(self.dataset_manager.processed_styles_path / "augmented"),
                num_augmentations=2
            )
            result["augmented"] = augmented_files
        
        return result
    
    def batch_preprocess(self, input_dir: str, dataset_type: str, 
                        remove_bg: bool = True, augment: bool = False) -> Dict:
        """Batch preprocess all images in a directory"""
        input_path = Path(input_dir)
        results = []
        
        for image_file in tqdm(input_path.glob("*"), desc=f"Processing {dataset_type}"):
            if not image_file.is_file():
                continue
            
            try:
                if dataset_type == "garments":
                    result = self.preprocess_garment(str(image_file), remove_bg, augment)
                elif dataset_type == "styles":
                    result = self.preprocess_style(str(image_file), augment)
                else:
                    raise ValueError(f"Invalid dataset type: {dataset_type}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        return {
            "processed_count": len(results),
            "results": results,
            "dataset_stats": self.dataset_manager.get_dataset_stats()
        }
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of preprocessing pipeline"""
        return {
            "target_size": self.target_size,
            "device": self.device,
            "dataset_stats": self.dataset_manager.get_dataset_stats(),
            "available_augmentations": list(self.augmentation.transform_pipeline.keys()),
            "background_removal_methods": ["threshold", "grabcut"]
        }

def main():
    """Example usage of the preprocessing pipeline"""
    pipeline = PreprocessingPipeline()
    
    # Example: Add some sample images to the dataset
    print("Dataset & Preprocessing Pipeline")
    print("=" * 40)
    
    # Show current stats
    stats = pipeline.dataset_manager.get_dataset_stats()
    print(f"Current dataset stats: {stats}")
    
    # Show preprocessing summary
    summary = pipeline.get_preprocessing_summary()
    print(f"Preprocessing summary: {summary}")

if __name__ == "__main__":
    main()
