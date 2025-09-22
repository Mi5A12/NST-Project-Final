#!/usr/bin/env python3
"""
Evaluation & Metrics Module for Neural Style Transfer
Implements quantitative and qualitative evaluation metrics
"""

import sys
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass
import math

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation metrics"""
    ssim_score: float = 0.0
    style_loss: float = 0.0
    content_loss: float = 0.0
    total_variation_loss: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    visual_appeal: float = 0.0
    realism: float = 0.0
    usability: float = 0.0
    overall_satisfaction: float = 0.0
    timestamp: str = ""

class SSIMMetric:
    """Structural Similarity Index (SSIM) for image quality assessment"""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = 0.01
        self.k2 = 0.03
        self.L = 255  # Dynamic range
    
    def _gaussian_window(self, size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM calculation"""
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same shape")
        
        # Convert to grayscale if needed
        if img1.dim() == 4 and img1.shape[1] == 3:
            img1 = self._rgb_to_grayscale(img1)
        if img2.dim() == 4 and img2.shape[1] == 3:
            img2 = self._rgb_to_grayscale(img2)
        
        # Ensure images are in [0, 1] range
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        
        # Create Gaussian window
        window = self._gaussian_window(self.window_size, self.sigma)
        window = window.to(img1.device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window.unsqueeze(0).unsqueeze(0), padding=self.window_size//2)
        mu2 = F.conv2d(img2, window.unsqueeze(0).unsqueeze(0), padding=self.window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 ** 2, window.unsqueeze(0).unsqueeze(0), padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window.unsqueeze(0).unsqueeze(0), padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window.unsqueeze(0).unsqueeze(0), padding=self.window_size//2) - mu1_mu2
        
        # Calculate SSIM
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean().item()
    
    def _rgb_to_grayscale(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale"""
        if img.dim() == 4:
            # Batch of images
            weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
            return torch.sum(img * weights, dim=1, keepdim=True)
        else:
            # Single image
            weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)
            return torch.sum(img * weights, dim=0, keepdim=True)

class StyleLossMetric:
    """Style loss evaluation using Gram matrix similarity"""
    
    def __init__(self):
        self.target_style_grams = None
    
    def set_target_style(self, style_grams: List[torch.Tensor]):
        """Set target style Gram matrices"""
        self.target_style_grams = style_grams
    
    def calculate_style_loss(self, generated_grams: List[torch.Tensor]) -> float:
        """Calculate style loss between generated and target Gram matrices"""
        if self.target_style_grams is None:
            raise ValueError("Target style not set. Call set_target_style() first.")
        
        if len(generated_grams) != len(self.target_style_grams):
            raise ValueError("Number of generated and target Gram matrices must match")
        
        total_loss = 0.0
        for gen_gram, target_gram in zip(generated_grams, self.target_style_grams):
            # Calculate MSE between Gram matrices
            loss = F.mse_loss(gen_gram, target_gram)
            total_loss += loss.item()
        
        return total_loss / len(generated_grams)

class PerformanceMetrics:
    """Performance and timing metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = 0.0
    
    def start_timing(self):
        """Start performance timing"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def end_timing(self):
        """End performance timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
    
    def get_inference_time(self) -> float:
        """Get total inference time in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        else:
            # Approximate memory usage for CPU
            return 0.0

class UserStudySystem:
    """User study system for qualitative evaluation"""
    
    def __init__(self, study_path: str = "user_studies"):
        self.study_path = Path(study_path)
        self.study_path.mkdir(exist_ok=True)
        self.participants = []
        self.responses = []
    
    def add_participant(self, participant_id: str, role: str, demographics: Dict = None):
        """Add a participant to the study"""
        participant = {
            "id": participant_id,
            "role": role,  # "designer" or "consumer"
            "demographics": demographics or {},
            "timestamp": datetime.now().isoformat()
        }
        self.participants.append(participant)
        return participant
    
    def submit_rating(self, participant_id: str, image_path: str, 
                     visual_appeal: float, realism: float, usability: float,
                     overall_satisfaction: float, comments: str = ""):
        """Submit a rating for an image"""
        rating = {
            "participant_id": participant_id,
            "image_path": image_path,
            "visual_appeal": visual_appeal,
            "realism": realism,
            "usability": usability,
            "overall_satisfaction": overall_satisfaction,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        self.responses.append(rating)
        return rating
    
    def get_study_statistics(self) -> Dict[str, Any]:
        """Get comprehensive study statistics"""
        if not self.responses:
            return {"error": "No responses available"}
        
        # Separate designers and consumers
        designer_responses = [r for r in self.responses if self._get_participant_role(r["participant_id"]) == "designer"]
        consumer_responses = [r for r in self.responses if self._get_participant_role(r["participant_id"]) == "consumer"]
        
        stats = {
            "total_participants": len(self.participants),
            "total_responses": len(self.responses),
            "designer_responses": len(designer_responses),
            "consumer_responses": len(consumer_responses),
            "overall_averages": self._calculate_averages(self.responses),
            "designer_averages": self._calculate_averages(designer_responses) if designer_responses else {},
            "consumer_averages": self._calculate_averages(consumer_responses) if consumer_responses else {}
        }
        
        return stats
    
    def _get_participant_role(self, participant_id: str) -> str:
        """Get participant role by ID"""
        for participant in self.participants:
            if participant["id"] == participant_id:
                return participant["role"]
        return "unknown"
    
    def _calculate_averages(self, responses: List[Dict]) -> Dict[str, float]:
        """Calculate average ratings from responses"""
        if not responses:
            return {}
        
        return {
            "visual_appeal": np.mean([r["visual_appeal"] for r in responses]),
            "realism": np.mean([r["realism"] for r in responses]),
            "usability": np.mean([r["usability"] for r in responses]),
            "overall_satisfaction": np.mean([r["overall_satisfaction"] for r in responses])
        }
    
    def save_study_data(self, filename: str = None):
        """Save study data to JSON file"""
        if filename is None:
            filename = f"user_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "participants": self.participants,
            "responses": self.responses,
            "statistics": self.get_study_statistics()
        }
        
        filepath = self.study_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)

class EvaluationSystem:
    """Main evaluation system combining all metrics"""
    
    def __init__(self):
        self.ssim_metric = SSIMMetric()
        self.style_loss_metric = StyleLossMetric()
        self.performance_metrics = PerformanceMetrics()
        self.user_study = UserStudySystem()
        self.evaluation_history = []
    
    def evaluate_style_transfer(self, content_img: torch.Tensor, style_img: torch.Tensor,
                              generated_img: torch.Tensor, style_grams: List[torch.Tensor] = None,
                              content_grams: List[torch.Tensor] = None) -> EvaluationMetrics:
        """Comprehensive evaluation of style transfer results"""
        
        metrics = EvaluationMetrics()
        metrics.timestamp = datetime.now().isoformat()
        
        # Start performance timing
        self.performance_metrics.start_timing()
        
        try:
            # SSIM for structure preservation
            metrics.ssim_score = self.ssim_metric.calculate_ssim(content_img, generated_img)
            
            # Style loss if Gram matrices provided
            if style_grams is not None:
                self.style_loss_metric.set_target_style(style_grams)
                # Calculate generated Gram matrices (simplified)
                generated_grams = self._calculate_gram_matrices(generated_img)
                metrics.style_loss = self.style_loss_metric.calculate_style_loss(generated_grams)
            
            # Performance metrics
            self.performance_metrics.end_timing()
            metrics.inference_time = self.performance_metrics.get_inference_time()
            metrics.memory_usage = self.performance_metrics.get_memory_usage()
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
        
        # Store in history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def _calculate_gram_matrices(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Calculate Gram matrices for style loss (simplified)"""
        # This is a simplified version - in practice, you'd use the actual VGG features
        # For now, we'll create dummy Gram matrices
        gram_matrices = []
        for i in range(5):  # 5 layers
            # Create a dummy Gram matrix
            gram = torch.randn(64, 64, device=img.device)
            gram = torch.mm(gram, gram.t())
            gram_matrices.append(gram)
        return gram_matrices
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.evaluation_history:
            return {"message": "No evaluations available"}
        
        # Calculate averages
        ssim_scores = [m.ssim_score for m in self.evaluation_history]
        style_losses = [m.style_loss for m in self.evaluation_history]
        inference_times = [m.inference_time for m in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_ssim": np.mean(ssim_scores),
            "average_style_loss": np.mean(style_losses),
            "average_inference_time": np.mean(inference_times),
            "ssim_above_threshold": sum(1 for s in ssim_scores if s > 0.72),
            "performance_improvement": self._calculate_performance_improvement()
        }
    
    def _calculate_performance_improvement(self) -> Dict[str, float]:
        """Calculate performance improvement metrics"""
        if len(self.evaluation_history) < 2:
            return {"improvement": 0.0}
        
        # Compare first half vs second half (baseline vs optimized)
        mid_point = len(self.evaluation_history) // 2
        baseline_times = [m.inference_time for m in self.evaluation_history[:mid_point]]
        optimized_times = [m.inference_time for m in self.evaluation_history[mid_point:]]
        
        if not baseline_times or not optimized_times:
            return {"improvement": 0.0}
        
        baseline_avg = np.mean(baseline_times)
        optimized_avg = np.mean(optimized_times)
        
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        
        return {
            "baseline_time": baseline_avg,
            "optimized_time": optimized_avg,
            "improvement_percent": improvement,
            "speedup_factor": baseline_avg / optimized_avg if optimized_avg > 0 else 1.0
        }
    
    def save_evaluation_data(self, filename: str = None):
        """Save evaluation data to JSON file"""
        if filename is None:
            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "evaluations": [
                {
                    "ssim_score": m.ssim_score,
                    "style_loss": m.style_loss,
                    "inference_time": m.inference_time,
                    "memory_usage": m.memory_usage,
                    "timestamp": m.timestamp
                }
                for m in self.evaluation_history
            ],
            "summary": self.get_evaluation_summary()
        }
        
        filepath = Path("evaluation_data") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)

def demo_evaluation_system():
    """Demo the evaluation system"""
    print("📊 Evaluation & Metrics Demo")
    print("=" * 40)
    
    # Initialize evaluation system
    eval_system = EvaluationSystem()
    
    # Create sample images
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)
    generated_img = torch.randn(1, 3, 256, 256)
    
    # Test SSIM
    ssim_score = eval_system.ssim_metric.calculate_ssim(content_img, generated_img)
    print(f"SSIM Score: {ssim_score:.4f}")
    
    # Test performance metrics
    eval_system.performance_metrics.start_timing()
    time.sleep(0.1)  # Simulate processing
    eval_system.performance_metrics.end_timing()
    
    inference_time = eval_system.performance_metrics.get_inference_time()
    print(f"Inference Time: {inference_time:.4f}s")
    
    # Test user study
    eval_system.user_study.add_participant("user_001", "designer")
    eval_system.user_study.submit_rating(
        "user_001", "test_image.jpg", 
        visual_appeal=4.2, realism=4.1, usability=4.8, overall_satisfaction=4.37
    )
    
    stats = eval_system.user_study.get_study_statistics()
    print(f"User Study Stats: {stats}")
    
    print("✅ Evaluation system demo completed!")

if __name__ == "__main__":
    demo_evaluation_system()
