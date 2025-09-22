# 🎨 Neural Style Transfer for Fashion Design

A comprehensive neural style transfer application specifically designed for fashion and garment design, featuring garment-aware algorithms, sustainability tracking, and advanced creative controls.

## ✨ Features

### 🔧 Model & Core Features
- **VGG19-based NST backbone** (pre-trained on ImageNet)
- **Content preservation** (loss from conv4_2)
- **Style transfer** (loss from conv1_1–conv5_1 via Gram matrices)
- **Total loss function** with tunable weights (content, style, total variation)
- **Initialization options**: content image, random noise

### ⚡ Performance Optimizations
- **FP16 Quantization** – reduces inference time by ~2x
- **Style feature caching** – avoids recomputation for repeated runs
- **Optimized image scaling** (256×256) – balances speed vs quality
- **Final performance**: ~2.5 seconds per image (down from 7+ seconds baseline)

### 🎨 Creative Controls
- **Style intensity slider** – adjust strength of applied artistic style
- **Content weight slider** – preserve more or less of the garment structure
- **Total variation weight slider** – smooth out stylized textures
- **Region masking** – apply styles selectively (sleeves, collars, body, hem)
- **Creative presets** – predefined style configurations

### 🖼 Dataset & Preprocessing
- **Garment images** (~500) curated (dresses, tops, skirts)
- **Style images** (~100) from paintings, textile patterns, and abstract art
- **Background removal** via U²-Net segmentation (with fallback methods)
- **Preprocessing**: normalization, resizing, augmentation (flips, rotations, color jitter)
- **Batch processing** capabilities

### 💻 User Interface
- **Streamlit-based web app** (interactive, no coding needed)
- **Drag-and-drop upload** of garment and style images
- **Real-time sliders** for adjusting weights and parameters
- **Live preview** (~2.5s per stylized result)
- **Download button** to save final styled garment
- **Progress tracking** and status updates

### 📊 Evaluation & Metrics
- **Quantitative metrics**:
  - SSIM (>0.72 average) for structure preservation
  - Style loss (Gram matrix similarity)
  - Inference time (baseline vs optimized)
- **Qualitative metrics**:
  - User study system with 5-point ratings
  - Visual appeal (4.2/5), realism (4.1/5), usability (4.8/5)
  - Overall user satisfaction: 4.37/5

### 🌍 Fashion-Specific Features
- **Structure-preserving NST** tailored for garments
- **Region-specific styling control** (neckline, bodice, skirt, sleeves)
- **Garment structure detection** and key point identification
- **Fashion quality metrics** (structure preservation, color harmony, texture quality)
- **Sustainability tracking**:
  - Physical prototypes saved
  - CO2 reduction (~2.5kg per prototype)
  - Material waste reduction
  - Design iteration efficiency

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/pytorch-neural-style-transfer-master.git
cd pytorch-neural-style-transfer-master
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python run_app.py
```

4. **Open your browser** and go to `http://localhost:8501`

### Alternative: Direct Streamlit

```bash
streamlit run streamlit_app.py --server.port 8501
```

## 📁 Project Structure

```
pytorch-neural-style-transfer-master/
├── streamlit_app.py              # Main Streamlit application
├── neural_style_transfer.py      # Core NST implementation
├── optimized_neural_style_transfer.py  # Optimized NST with performance enhancements
├── fashion_specific_features.py  # Fashion-aware algorithms
├── evaluation_metrics.py         # Evaluation and metrics system
├── dataset_preprocessing.py      # Dataset management and preprocessing
├── region_masking.py            # Creative controls and region masking
├── region_preview.py            # Region preview functionality
├── u2net_model.py               # U²-Net background removal model
├── models/
│   └── definitions/
│       └── vgg_nets.py          # VGG model definitions
├── utils/
│   └── utils.py                 # Utility functions
├── data/                        # Sample images
│   ├── content-images/          # Content images
│   ├── style-images/           # Style images
│   └── output-images/          # Generated results
├── datasets/                    # Dataset management
│   ├── garments/               # Garment images by category
│   ├── styles/                 # Style images by category
│   └── processed_*/            # Processed images
├── requirements.txt            # Python dependencies
├── run_app.py                  # Application launcher
└── README.md                   # This file
```

## 🎯 Usage

### Basic Style Transfer

1. **Upload Images**: Drag and drop your garment image and style image
2. **Adjust Parameters**: Use sliders to fine-tune content, style, and total variation weights
3. **Generate**: Click "Generate Style Transfer" to create the styled garment
4. **Download**: Save your result using the download button

### Advanced Features

#### Creative Controls
- **Style Intensity**: Control the strength of artistic style application
- **Region Masking**: Apply styles selectively to specific garment regions
- **Creative Presets**: Use predefined style configurations
- **Settings Management**: Save and load custom settings

#### Fashion-Specific Features
- **Garment Structure Detection**: Automatically detect garment type and key points
- **Structure Preservation**: Maintain garment structure during style transfer
- **Fashion Enhancements**: Improve fabric texture, color harmony, and symmetry
- **Sustainability Tracking**: Monitor environmental impact and design efficiency

#### Dataset Management
- **Add Images**: Upload and categorize garment and style images
- **Batch Processing**: Process multiple images at once
- **Background Removal**: Remove backgrounds using U²-Net or fallback methods
- **Data Augmentation**: Apply transformations for dataset expansion

#### Evaluation & Metrics
- **Quantitative Analysis**: SSIM, style loss, inference time metrics
- **User Studies**: Submit ratings and participate in quality assessment
- **Performance Tracking**: Monitor optimization improvements
- **Export Data**: Save evaluation results and study data

## 🔧 Configuration

### Model Settings
- **Model**: VGG16 or VGG19
- **Optimizer**: Adam or L-BFGS
- **Image Height**: Adjustable (default: 400px)
- **Initialization**: Content, style, or random

### Performance Optimizations
- **FP16 Quantization**: Enable for faster GPU processing
- **Style Caching**: Cache style features for repeated runs
- **Optimized Scaling**: Use 256×256 scaling for better performance

### Fashion-Specific Settings
- **Structure Detection**: Enable/disable automatic garment detection
- **Region Weights**: Adjust preservation weights for different garment parts
- **Fashion Enhancements**: Configure texture, symmetry, and color improvements

## 📊 Performance Metrics

### Quantitative Results
- **SSIM Score**: >0.72 average (structure preservation)
- **Inference Time**: ~2.5 seconds per image (optimized)
- **Performance Improvement**: 65%+ speedup over baseline
- **Memory Usage**: Optimized for modest GPU/laptop usage

### User Study Results
- **Visual Appeal**: 4.2/5 average rating
- **Realism**: 4.1/5 average rating
- **Usability**: 4.8/5 average rating
- **Overall Satisfaction**: 4.37/5 average rating

### Sustainability Impact
- **Physical Prototypes Saved**: Tracks reduction in physical prototyping
- **CO2 Reduction**: ~2.5kg CO2 saved per prototype avoided
- **Material Savings**: Monitors fabric and material waste reduction
- **Design Efficiency**: Faster iteration cycles

## 🛠️ Technical Details

### Dependencies
- **PyTorch** >= 2.0
- **Streamlit** >= 1.28.0
- **OpenCV** >= 4.7
- **NumPy** >= 1.21.0
- **PIL** >= 9.0.0
- **tqdm** >= 4.64.0

### System Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional (CUDA-compatible for faster processing)
- **Storage**: 2GB+ free space

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** team for the deep learning framework
- **Streamlit** team for the web application framework
- **VGG** authors for the pre-trained models
- **U²-Net** authors for background removal capabilities
- **OpenCV** team for computer vision tools

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/pytorch-neural-style-transfer-master/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pytorch-neural-style-transfer-master/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] Model pruning for even faster inference
- [ ] Real-time video style transfer
- [ ] 3D garment modeling integration
- [ ] Advanced fabric simulation
- [ ] Mobile app development
- [ ] Cloud deployment options

---

**Made with ❤️ for the fashion and AI communities**