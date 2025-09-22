# Neural Style Transfer with Streamlit

A modern, interactive web application for neural style transfer using PyTorch and Streamlit. Transform your images with artistic styles using deep learning.

## Features

- 🎨 **Interactive UI**: Clean, modern interface built with Streamlit
- 🖼️ **Image Upload**: Easy drag-and-drop image upload
- ⚙️ **Customizable Parameters**: Adjust content weight, style weight, and other parameters
- 🚀 **Real-time Processing**: Live progress tracking during style transfer
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 💾 **Download Results**: Save your generated images

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pytorch-neural-style-transfer-master
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## Usage

1. **Upload Images**: 
   - Upload a content image (the image you want to style)
   - Upload a style image (the artistic style to apply)

2. **Configure Parameters**:
   - **Model**: Choose between VGG16 or VGG19 (VGG19 recommended)
   - **Optimizer**: Select L-BFGS (faster) or Adam
   - **Image Height**: Set resolution (256-1024 pixels)
   - **Initialization**: Choose how to start the optimization
   - **Weights**: Adjust content, style, and total variation weights

3. **Generate**: Click "Generate Style Transfer" and wait for processing

4. **Download**: Save your result when complete

## Parameters Explained

- **Content Weight**: Higher values preserve more of the original image structure
- **Style Weight**: Higher values apply more of the artistic style
- **Total Variation Weight**: Higher values reduce noise and artifacts
- **Initialization Method**:
  - **Content**: Start with the content image (recommended)
  - **Style**: Start with the style image
  - **Random**: Start with random noise

## Technical Details

- **Models**: VGG16 and VGG19 pre-trained on ImageNet
- **Optimization**: L-BFGS or Adam optimizer
- **Framework**: PyTorch for neural networks, Streamlit for UI
- **Processing**: GPU acceleration when available

## File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── neural_style_transfer.py  # Core neural style transfer logic
├── utils/                    # Utility functions
│   ├── utils.py             # Image processing utilities
│   └── video_utils.py       # Video creation utilities
├── models/                   # Neural network definitions
│   └── definitions/
│       └── vgg_nets.py      # VGG model implementations
├── data/                     # Sample images
│   ├── content-images/      # Content images
│   ├── style-images/        # Style images
│   └── output-images/       # Generated results
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- OpenCV
- NumPy
- Pillow

## Troubleshooting

- **CUDA Issues**: The app will automatically use CPU if CUDA is not available
- **Memory Issues**: Try reducing the image height parameter
- **Slow Processing**: Use L-BFGS optimizer and lower resolution for faster results

## License

This project is based on the original neural style transfer implementation by Gatys et al. (2016).

## Acknowledgments

- Original paper: "Image Style Transfer Using Convolutional Neural Networks" by Gatys et al.
- PyTorch implementation inspired by various open-source projects
- UI built with Streamlit
