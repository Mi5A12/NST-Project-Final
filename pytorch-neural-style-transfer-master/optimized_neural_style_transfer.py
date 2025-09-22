import sys
import warnings
import time
import os
import tempfile
import shutil
from typing import Dict, Tuple, Optional
import hashlib

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2

import utils.utils as utils
from models.definitions.vgg_nets import Vgg16, Vgg19


class OptimizedNeuralStyleTransfer:
    """Optimized Neural Style Transfer with performance improvements"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.style_cache = {}  # Cache for style features
        self.model_cache = {}  # Cache for loaded models
        self.performance_metrics = {
            'model_loading_time': 0,
            'style_processing_time': 0,
            'content_processing_time': 0,
            'optimization_time': 0,
            'total_time': 0
        }
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _get_model_hash(self, model_name: str) -> str:
        """Generate hash for model configuration"""
        return hashlib.md5(f"{model_name}_{self.device}".encode()).hexdigest()
    
    def _get_style_hash(self, style_path: str, model_name: str) -> str:
        """Generate hash for style image and model"""
        stat = os.stat(style_path)
        return hashlib.md5(f"{style_path}_{stat.st_size}_{stat.st_mtime}_{model_name}".encode()).hexdigest()
    
    def load_model(self, model_name: str, use_quantization: bool = True):
        """Load and cache model with optimizations"""
        start_time = time.time()
        
        model_hash = self._get_model_hash(model_name)
        if model_hash in self.model_cache:
            self.performance_metrics['model_loading_time'] = 0.001  # Cache hit
            return self.model_cache[model_hash]
        
        # Load model
        if model_name == 'vgg16':
            model = Vgg16(requires_grad=False, show_progress=False)
        elif model_name == 'vgg19':
            model = Vgg19(requires_grad=False, show_progress=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(self.device)
        
        # Apply optimizations (only on GPU)
        if use_quantization and torch.cuda.is_available():
            # Use FP16 for faster inference on GPU
            model = model.half()
        else:
            # Ensure float32 for CPU compatibility
            model = model.float()
        
        # Skip torch.compile on macOS due to library loading issues
        # This is a known issue with PyTorch 2.0+ on macOS
        pass
        
        model.eval()
        
        # Cache the model
        self.model_cache[model_hash] = model
        
        self.performance_metrics['model_loading_time'] = time.time() - start_time
        return model
    
    def preprocess_image_optimized(self, img_path: str, target_size: int = 256) -> torch.Tensor:
        """Optimized image preprocessing with smart scaling"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Smart scaling - maintain aspect ratio but optimize for performance
        h, w = img.shape[:2]
        
        # Scale to target size while maintaining aspect ratio
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        # Use high-quality interpolation
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = img_tensor.mul(255)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Apply quantization if needed (only on GPU)
        if (hasattr(self, '_current_model') and hasattr(self._current_model, 'dtype') 
            and self._current_model.dtype == torch.float16 and torch.cuda.is_available()):
            img_tensor = img_tensor.half()
        else:
            # Ensure float32 for CPU compatibility
            img_tensor = img_tensor.float()
        
        return img_tensor
    
    def get_style_features_cached(self, style_path: str, model_name: str, model) -> Tuple[torch.Tensor, list]:
        """Get style features with caching"""
        start_time = time.time()
        
        style_hash = self._get_style_hash(style_path, model_name)
        if style_hash in self.style_cache:
            self.performance_metrics['style_processing_time'] = 0.001  # Cache hit
            return self.style_cache[style_hash]
        
        # Process style image
        style_img = self.preprocess_image_optimized(style_path)
        
        # Get style features
        with torch.no_grad():
            style_features = model(style_img)
            style_grams = [utils.gram_matrix(x) for x in style_features]
        
        # Cache the results
        self.style_cache[style_hash] = (style_img, style_grams)
        
        self.performance_metrics['style_processing_time'] = time.time() - start_time
        return style_img, style_grams
    
    def get_content_features(self, content_path: str, model) -> torch.Tensor:
        """Get content features"""
        start_time = time.time()
        
        content_img = self.preprocess_image_optimized(content_path)
        
        with torch.no_grad():
            content_features = model(content_img)
            content_feature = content_features[0].squeeze(0)  # conv4_2 for VGG19
        
        self.performance_metrics['content_processing_time'] = time.time() - start_time
        return content_img, content_feature
    
    def build_loss_optimized(self, model, optimizing_img, content_feature, style_grams, config):
        """Optimized loss computation"""
        with torch.enable_grad():
            current_features = model(optimizing_img)
            
            # Content loss
            current_content = current_features[0].squeeze(0)
            content_loss = nn.MSELoss(reduction='mean')(content_feature, current_content)
            
            # Style loss
            style_loss = 0.0
            current_style_grams = [utils.gram_matrix(x) for x in current_features]
            
            for gram_gt, gram_hat in zip(style_grams, current_style_grams):
                style_loss += nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            
            style_loss /= len(style_grams)
            
            # Total variation loss
            tv_loss = utils.total_variation(optimizing_img)
            
            # Total loss
            total_loss = (config['content_weight'] * content_loss + 
                         config['style_weight'] * style_loss + 
                         config['tv_weight'] * tv_loss)
            
            return total_loss, content_loss, style_loss, tv_loss
    
    def optimize_image(self, model, content_img, style_grams, content_feature, config, 
                      progress_callback=None) -> torch.Tensor:
        """Optimized image optimization with progress tracking"""
        start_time = time.time()
        
        # Initialize optimizing image as a proper leaf tensor
        if config['init_method'] == 'content':
            # Detach and clone to create a proper leaf tensor
            optimizing_img = content_img.detach().clone().requires_grad_(True)
        elif config['init_method'] == 'style':
            # Use style image as initialization
            style_img = self.preprocess_image_optimized(config.get('style_path', ''))
            optimizing_img = style_img.detach().clone().requires_grad_(True)
        else:  # random
            optimizing_img = torch.randn_like(content_img).requires_grad_(True)
        
        # Ensure tensor compatibility for L-BFGS optimizer
        if config['optimizer'] == 'lbfgs':
            # L-BFGS works better with float32 tensors
            if optimizing_img.dtype == torch.float16:
                optimizing_img = optimizing_img.float()
            # Ensure contiguous memory layout and proper leaf tensor
            optimizing_img = optimizing_img.detach().contiguous().requires_grad_(True)
        
        # Setup optimizer
        if config['optimizer'] == 'adam':
            optimizer = Adam([optimizing_img], lr=1e1)
            num_iterations = 1000  # Reduced for faster processing
        else:  # lbfgs
            optimizer = LBFGS([optimizing_img], max_iter=50, line_search_fn='strong_wolfe')
            num_iterations = 50
        
        # Optimization loop
        for iteration in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                total_loss, content_loss, style_loss, tv_loss = self.build_loss_optimized(
                    model, optimizing_img, content_feature, style_grams, config
                )
                total_loss.backward()
                return total_loss
            
            if config['optimizer'] == 'adam':
                optimizer.step(closure)
            else:
                optimizer.step(closure)
            
            # Progress callback
            if progress_callback and iteration % 10 == 0:
                with torch.no_grad():
                    total_loss, content_loss, style_loss, tv_loss = self.build_loss_optimized(
                        model, optimizing_img, content_feature, style_grams, config
                    )
                    progress_callback(iteration, total_loss.item(), content_loss.item(), 
                                    style_loss.item(), tv_loss.item())
        
        self.performance_metrics['optimization_time'] = time.time() - start_time
        return optimizing_img
    
    def process_style_transfer(self, content_path: str, style_path: str, config: Dict, 
                             progress_callback=None) -> Tuple[torch.Tensor, Dict]:
        """Main style transfer processing with optimizations"""
        total_start = time.time()
        
        # Load model
        model = self.load_model(config['model'], use_quantization=config.get('use_quantization', True))
        self._current_model = model
        
        # Get features
        content_img, content_feature = self.get_content_features(content_path, model)
        style_img, style_grams = self.get_style_features_cached(style_path, config['model'], model)
        
        # Ensure all input tensors are detached and don't require gradients
        content_img = content_img.detach()
        style_img = style_img.detach()
        
        # Store style path for initialization
        config['style_path'] = style_path
        
        # Optimize image
        result_img = self.optimize_image(model, content_img, style_grams, content_feature, 
                                       config, progress_callback)
        
        self.performance_metrics['total_time'] = time.time() - total_start
        
        return result_img, self.performance_metrics
    
    def save_result(self, result_img: torch.Tensor, output_path: str, config: Dict):
        """Save result image with proper formatting"""
        with torch.no_grad():
            # Convert to numpy
            if result_img.dtype == torch.float16:
                result_img = result_img.float()
            
            result_np = result_img.squeeze(0).cpu().numpy()
            result_np = np.moveaxis(result_np, 0, 2)
            
            # Denormalize
            result_np += np.array([123.675, 116.28, 103.53]).reshape((1, 1, 3))
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(output_path, result_bgr)
    
    def clear_cache(self):
        """Clear all caches to free memory"""
        self.style_cache.clear()
        self.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary"""
        return {
            'model_loading_time': f"{self.performance_metrics['model_loading_time']:.3f}s",
            'style_processing_time': f"{self.performance_metrics['style_processing_time']:.3f}s",
            'content_processing_time': f"{self.performance_metrics['content_processing_time']:.3f}s",
            'optimization_time': f"{self.performance_metrics['optimization_time']:.3f}s",
            'total_time': f"{self.performance_metrics['total_time']:.3f}s",
            'cache_hits': len(self.style_cache),
            'model_cache_size': len(self.model_cache)
        }


def optimized_neural_style_transfer(config: Dict) -> str:
    """Optimized neural style transfer function"""
    # Create optimizer instance
    optimizer = OptimizedNeuralStyleTransfer()
    
    # Setup paths
    content_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    
    out_dir_name = f"optimized_{os.path.split(content_path)[1].split('.')[0]}_{os.path.split(style_path)[1].split('.')[0]}"
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    
    # Progress callback for logging
    def progress_callback(iteration, total_loss, content_loss, style_loss, tv_loss):
        print(f"Optimization | iteration: {iteration:03}, total loss={total_loss:12.4f}, "
              f"content_loss={config['content_weight'] * content_loss:12.4f}, "
              f"style loss={config['style_weight'] * style_loss:12.4f}, "
              f"tv loss={config['tv_weight'] * tv_loss:12.4f}")
    
    # Process style transfer
    result_img, metrics = optimizer.process_style_transfer(
        content_path, style_path, config, progress_callback
    )
    
    # Save result
    output_filename = f"optimized_result_{config['model']}_{config['optimizer']}.jpg"
    output_path = os.path.join(dump_path, output_filename)
    optimizer.save_result(result_img, output_path, config)
    
    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    for key, value in optimizer.get_performance_summary().items():
        print(f"{key}: {value}")
    print("="*50)
    
    # Cleanup
    optimizer.clear_cache()
    
    return dump_path
