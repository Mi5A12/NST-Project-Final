#!/usr/bin/env python3
"""
U²-Net Model Implementation for Background Removal
Based on the paper: "U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple

class REBNCONV(nn.Module):
    """ReBNConv block used in U²-Net"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class RSU4F(nn.Module):
    """RSU-4F block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU4F, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.img_size = img_size
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv2d = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv3d = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv4d = REBNCONV(mid_ch, mid_ch, dirate=16)
        self.rebnconv3u = REBNCONV(mid_ch*2, mid_ch, dirate=8)
        self.rebnconv2u = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv1u = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconvout = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx1d = self.rebnconv1d(hx1)
        hx2d = self.rebnconv2d(hx1d)
        hx3d = self.rebnconv3d(hx2d)
        hx4d = self.rebnconv4d(hx3d)
        hx3u = self.rebnconv3u(torch.cat((hx4d, hx3d), 1))
        hx2u = self.rebnconv2u(torch.cat((hx3u, hx2d), 1))
        hx1u = self.rebnconv1u(torch.cat((hx2u, hx1d), 1))
        hxout = self.rebnconvout(torch.cat((hx1u, hx1), 1))
        return hxout + hxin

class RSU4(nn.Module):
    """RSU-4 block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU4, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.img_size = img_size
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconvout = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hxout = self.rebnconvout(torch.cat((hx1d, hxin), 1))
        return hxout + hxin

class U2NET(nn.Module):
    """U²-Net model for salient object detection"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU4(in_ch, 64, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(64, 128, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(128, 256, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 512, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 512, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 512, 512)
        
        # Decoder
        self.stage5d = RSU4F(1024, 512, 512)
        self.stage4d = RSU4(1024, 256, 256)
        self.stage3d = RSU4(512, 128, 128)
        self.stage2d = RSU4(256, 64, 64)
        self.stage1d = RSU4(128, 64, 64)
        
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        # Final output
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        
        # Decoder
        hx6d = self.stage5d(torch.cat((hx6, hx5), 1))
        hx6dup = F.interpolate(hx6d, size=(hx4.size(2), hx4.size(3)), mode='bilinear', align_corners=True)
        hx5d = self.stage4d(torch.cat((hx6dup, hx4), 1))
        hx5dup = F.interpolate(hx5d, size=(hx3.size(2), hx3.size(3)), mode='bilinear', align_corners=True)
        hx4d = self.stage3d(torch.cat((hx5dup, hx3), 1))
        hx4dup = F.interpolate(hx4d, size=(hx2.size(2), hx2.size(3)), mode='bilinear', align_corners=True)
        hx3d = self.stage2d(torch.cat((hx4dup, hx2), 1))
        hx3dup = F.interpolate(hx3d, size=(hx1.size(2), hx1.size(3)), mode='bilinear', align_corners=True)
        hx2d = self.stage1d(torch.cat((hx3dup, hx1), 1))
        
        # Side outputs
        d1 = self.side1(hx2d)
        d2 = self.side2(hx3d)
        d3 = self.side3(hx4d)
        d4 = self.side4(hx5d)
        d5 = self.side5(hx6d)
        d6 = self.side6(hx6)
        
        # Upsample side outputs
        d2 = F.interpolate(d2, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        d5 = F.interpolate(d5, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        d6 = F.interpolate(d6, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        
        # Final output
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

class U2NETPredictor:
    """U²-Net predictor for background removal"""
    
    def __init__(self, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = U2NET(in_ch=3, out_ch=1)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded U²-Net model from {model_path}")
        else:
            print("U²-Net model not found, using random weights")
        
        self.model.to(device)
        self.model.eval()
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 320) -> torch.Tensor:
        """Preprocess image for U²-Net"""
        # Resize image
        image = cv2.resize(image, (target_size, target_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_mask(self, mask: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess mask to original image size"""
        # Convert to numpy
        mask = mask.squeeze().cpu().numpy()
        
        # Resize to original size
        mask = cv2.resize(mask, original_size)
        
        # Threshold to binary mask
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        return mask
    
    def remove_background(self, image: np.ndarray, target_size: int = 320) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background using U²-Net"""
        original_size = (image.shape[1], image.shape[0])
        
        # Preprocess
        input_tensor = self.preprocess_image(image, target_size)
        
        # Predict
        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(input_tensor)
        
        # Use the final output
        mask = self.postprocess_mask(d0, original_size)
        
        # Apply mask to image
        result = image.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = mask
        
        return result, mask

def download_u2net_weights(url: str = None, save_path: str = "u2net_weights.pth"):
    """Download U²-Net weights (placeholder for actual implementation)"""
    if url is None:
        print("U²-Net weights not available for download")
        return False
    
    try:
        import requests
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded U²-Net weights to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download U²-Net weights: {e}")
        return False

if __name__ == "__main__":
    # Test U²-Net model
    model = U2NET(in_ch=3, out_ch=1)
    print(f"U²-Net model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model output shape: {output[0].shape}")
