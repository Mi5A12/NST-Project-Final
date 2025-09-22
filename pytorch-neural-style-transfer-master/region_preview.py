#!/usr/bin/env python3
"""
Region Preview Component for Visual Region Masking
Shows users which regions will be styled
"""

import sys
import warnings
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict

# Fix numpy recursion issues
sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore", category=UserWarning)

class RegionPreview:
    """Creates visual previews of region masking"""
    
    def __init__(self):
        self.colors = {
            "sleeves": (255, 0, 0),      # Red
            "collar": (0, 255, 0),       # Green
            "body": (0, 0, 255),         # Blue
            "hem": (255, 255, 0),        # Yellow
            "custom": (255, 0, 255)      # Magenta
        }
    
    def create_region_preview(self, image: np.ndarray, selected_regions: List[str], 
                            image_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Create a visual preview of selected regions"""
        # Resize image to preview size using PIL
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(image_size)
        preview_img = np.array(pil_img)
        
        # Create overlay
        overlay = preview_img.copy()
        
        # Draw each selected region using numpy
        for i, region in enumerate(selected_regions):
            color = self.colors.get(region, (128, 128, 128))
            
            if region == "sleeves":
                # Left sleeve
                overlay[50:200, 0:60] = color
                # Right sleeve
                overlay[50:200, 196:256] = color
            elif region == "collar":
                # Collar area
                overlay[0:80, 80:176] = color
                # Simple ellipse approximation
                overlay[20:60, 80:176] = color
            elif region == "body":
                # Body area
                overlay[80:220, 60:196] = color
            elif region == "hem":
                # Hem area
                overlay[200:256, 40:216] = color
            elif region == "custom":
                # Custom area (center)
                overlay[100:156, 100:156] = color
        
        # Blend with original image using numpy
        alpha = 0.3
        result = (preview_img * (1 - alpha) + overlay * alpha).astype(np.uint8)
        
        return result
    
    def create_region_legend(self, selected_regions: List[str]) -> str:
        """Create a text legend for selected regions"""
        legend_items = []
        for region in selected_regions:
            color_name = {
                "sleeves": "Red",
                "collar": "Green", 
                "body": "Blue",
                "hem": "Yellow",
                "custom": "Magenta"
            }.get(region, "Gray")
            legend_items.append(f"• {region.title()}: {color_name}")
        
        return "\n".join(legend_items) if legend_items else "No regions selected"
    
    def show_region_preview(self, image: np.ndarray, selected_regions: List[str]):
        """Display region preview in Streamlit"""
        if not selected_regions:
            st.info("No regions selected for styling")
            return
        
        # Create preview
        preview = self.create_region_preview(image, selected_regions)
        
        # Display preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(preview, caption="Region Preview", width='stretch')
        
        with col2:
            st.write("**Selected Regions:**")
            legend = self.create_region_legend(selected_regions)
            st.text(legend)
            
            # Show region statistics
            st.write("**Region Statistics:**")
            st.metric("Total Regions", len(selected_regions))
            st.metric("Coverage", f"{len(selected_regions) * 25}%")  # Rough estimate

def demo_region_preview():
    """Demo the region preview functionality"""
    print("🎯 Region Preview Demo")
    print("=" * 40)
    
    # Create a sample image
    sample_image = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some features to make it look like a garment
    cv2.rectangle(sample_image, (60, 80), (196, 220), (200, 200, 200), -1)  # Body
    cv2.rectangle(sample_image, (0, 50), (60, 200), (180, 180, 180), -1)    # Left sleeve
    cv2.rectangle(sample_image, (196, 50), (256, 200), (180, 180, 180), -1) # Right sleeve
    cv2.rectangle(sample_image, (80, 0), (176, 80), (220, 220, 220), -1)    # Collar
    
    # Test region preview
    preview = RegionPreview()
    test_regions = ["sleeves", "collar", "body"]
    
    result = preview.create_region_preview(sample_image, test_regions)
    print(f"Preview created with regions: {test_regions}")
    print(f"Preview shape: {result.shape}")
    
    legend = preview.create_region_legend(test_regions)
    print(f"Legend: {legend}")
    
    print("✅ Region preview demo completed!")

if __name__ == "__main__":
    demo_region_preview()
