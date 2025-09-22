#!/usr/bin/env python3
"""
Demo script for Evaluation & Metrics features
"""

import sys
import warnings
import torch
import numpy as np
from evaluation_metrics import EvaluationSystem, SSIMMetric, StyleLossMetric, UserStudySystem
import time

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

def demo_ssim_metric():
    """Demo SSIM metric functionality"""
    print("📊 SSIM Metric Demo")
    print("=" * 40)
    
    # Create sample images
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)
    generated_img = content_img * 0.8 + style_img * 0.2  # Blend for realistic test
    
    # Test SSIM
    ssim_metric = SSIMMetric()
    ssim_score = ssim_metric.calculate_ssim(content_img, generated_img)
    
    print(f"SSIM Score: {ssim_score:.4f}")
    print(f"Target (>0.72): {'✅ PASS' if ssim_score > 0.72 else '❌ FAIL'}")
    
    # Test with identical images
    identical_ssim = ssim_metric.calculate_ssim(content_img, content_img)
    print(f"Identical images SSIM: {identical_ssim:.4f} (should be ~1.0)")
    
    print("✅ SSIM metric demo completed!")

def demo_style_loss_metric():
    """Demo style loss metric functionality"""
    print("\n🎨 Style Loss Metric Demo")
    print("=" * 40)
    
    # Create sample Gram matrices
    style_grams = [torch.randn(64, 64) for _ in range(5)]
    generated_grams = [torch.randn(64, 64) for _ in range(5)]
    
    # Test style loss
    style_loss_metric = StyleLossMetric()
    style_loss_metric.set_target_style(style_grams)
    style_loss = style_loss_metric.calculate_style_loss(generated_grams)
    
    print(f"Style Loss: {style_loss:.4f}")
    print(f"Lower is better: {'✅ Good' if style_loss < 1.0 else '⚠️ High'}")
    
    # Test with identical Gram matrices
    identical_loss = style_loss_metric.calculate_style_loss(style_grams)
    print(f"Identical Gram matrices loss: {identical_loss:.4f} (should be ~0.0)")
    
    print("✅ Style loss metric demo completed!")

def demo_user_study_system():
    """Demo user study system functionality"""
    print("\n👥 User Study System Demo")
    print("=" * 40)
    
    # Initialize user study
    user_study = UserStudySystem()
    
    # Add participants
    participants = [
        ("designer_001", "designer", {"experience": "5+ years"}),
        ("designer_002", "designer", {"experience": "3+ years"}),
        ("consumer_001", "consumer", {"age": "25-35"}),
        ("consumer_002", "consumer", {"age": "35-45"}),
        ("consumer_003", "consumer", {"age": "18-25"})
    ]
    
    for participant_id, role, demographics in participants:
        user_study.add_participant(participant_id, role, demographics)
        print(f"Added {role}: {participant_id}")
    
    # Submit ratings
    ratings = [
        ("designer_001", "image1.jpg", 4.2, 4.1, 4.8, 4.37, "Great style transfer!"),
        ("designer_002", "image1.jpg", 4.0, 4.0, 4.5, 4.2, "Good but could be better"),
        ("consumer_001", "image1.jpg", 4.5, 4.3, 4.9, 4.6, "Love the result!"),
        ("consumer_002", "image1.jpg", 3.8, 3.9, 4.2, 4.0, "Nice but not perfect"),
        ("consumer_003", "image1.jpg", 4.7, 4.4, 4.8, 4.6, "Amazing!")
    ]
    
    for participant_id, image_path, visual_appeal, realism, usability, satisfaction, comments in ratings:
        user_study.submit_rating(participant_id, image_path, visual_appeal, realism, usability, satisfaction, comments)
        print(f"Submitted rating from {participant_id}: {satisfaction}/5")
    
    # Get statistics
    stats = user_study.get_study_statistics()
    print(f"\nStudy Statistics:")
    print(f"Total Participants: {stats['total_participants']}")
    print(f"Total Responses: {stats['total_responses']}")
    print(f"Designer Responses: {stats['designer_responses']}")
    print(f"Consumer Responses: {stats['consumer_responses']}")
    
    if "overall_averages" in stats:
        print(f"\nOverall Averages:")
        print(f"Visual Appeal: {stats['overall_averages']['visual_appeal']:.1f}/5")
        print(f"Realism: {stats['overall_averages']['realism']:.1f}/5")
        print(f"Usability: {stats['overall_averages']['usability']:.1f}/5")
        print(f"Overall Satisfaction: {stats['overall_averages']['overall_satisfaction']:.1f}/5")
    
    print("✅ User study system demo completed!")

def demo_evaluation_system():
    """Demo complete evaluation system"""
    print("\n📊 Complete Evaluation System Demo")
    print("=" * 40)
    
    # Initialize evaluation system
    eval_system = EvaluationSystem()
    
    # Create sample images
    content_img = torch.randn(1, 3, 256, 256)
    style_img = torch.randn(1, 3, 256, 256)
    generated_img = content_img * 0.7 + style_img * 0.3
    
    # Simulate multiple evaluations (baseline vs optimized)
    for i in range(5):
        # Simulate different processing times
        if i < 3:
            # Baseline (slower)
            time.sleep(0.1)
        else:
            # Optimized (faster)
            time.sleep(0.05)
        
        # Evaluate
        metrics = eval_system.evaluate_style_transfer(content_img, style_img, generated_img)
        print(f"Evaluation {i+1}: SSIM={metrics.ssim_score:.3f}, Time={metrics.inference_time:.2f}s")
    
    # Get summary
    summary = eval_system.get_evaluation_summary()
    print(f"\nEvaluation Summary:")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Average SSIM: {summary['average_ssim']:.3f}")
    print(f"Average Inference Time: {summary['average_inference_time']:.2f}s")
    print(f"SSIM Above Threshold: {summary['ssim_above_threshold']}")
    
    if "performance_improvement" in summary:
        perf = summary["performance_improvement"]
        print(f"Performance Improvement: {perf['improvement_percent']:.1f}%")
        print(f"Speedup Factor: {perf['speedup_factor']:.1f}x")
    
    print("✅ Complete evaluation system demo completed!")

def main():
    """Run all evaluation demos"""
    print("📊 Neural Style Transfer - Evaluation & Metrics Demo")
    print("=" * 60)
    
    try:
        # Run individual demos
        demo_ssim_metric()
        demo_style_loss_metric()
        demo_user_study_system()
        demo_evaluation_system()
        
        print("\n🎉 All evaluation demos completed successfully!")
        print("\n🚀 Ready to use the full evaluation and metrics system!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
