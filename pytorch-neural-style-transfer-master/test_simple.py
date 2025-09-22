#!/usr/bin/env python3
"""
Simple tests for Neural Style Transfer for Fashion Design
"""

import sys
import os
import warnings

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

def test_core_modules():
    """Test core modules without complex imports"""
    try:
        # Test basic imports
        import torch
        import numpy as np
        from PIL import Image
        print("✅ Basic dependencies imported successfully")
        
        # Test PyTorch
        x = torch.randn(1, 3, 256, 256)
        assert x.shape == (1, 3, 256, 256)
        print("✅ PyTorch working correctly")
        
        # Test NumPy
        y = np.random.rand(256, 256, 3)
        assert y.shape == (256, 256, 3)
        print("✅ NumPy working correctly")
        
        # Test PIL
        img = Image.new('RGB', (256, 256), color='red')
        assert img.size == (256, 256)
        print("✅ PIL working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Core modules error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        "streamlit_app.py",
        "neural_style_transfer.py",
        "optimized_neural_style_transfer.py",
        "fashion_specific_features.py",
        "evaluation_metrics.py",
        "dataset_preprocessing.py",
        "region_masking.py",
        "region_preview.py",
        "u2net_model.py",
        "requirements.txt",
        "README.md",
        "run_app.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = [
        "models",
        "models/definitions",
        "utils",
        "data",
        "data/content-images",
        "data/style-images",
        "data/output-images"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_requirements():
    """Test that requirements.txt is valid"""
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        # Check for key dependencies
        key_deps = ["torch", "streamlit", "opencv-python", "numpy", "Pillow"]
        missing_deps = []
        
        for dep in key_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies in requirements.txt: {missing_deps}")
            return False
        else:
            print("✅ Requirements.txt contains all key dependencies")
            return True
    except Exception as e:
        print(f"❌ Requirements test error: {e}")
        return False

def main():
    """Run all simple tests"""
    print("🧪 Running simple tests for Neural Style Transfer for Fashion Design")
    print("=" * 70)
    
    tests = [
        test_core_modules,
        test_file_structure,
        test_directory_structure,
        test_requirements,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple tests passed! The project structure is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
