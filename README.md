<div align="center">

# ESRGAN Image Upscaler 🖼️

**ESRGAN Image Upscaler** is a cutting-edge image enhancement platform built using advanced AI technologies. It includes an intuitive user interface, fast processing, and seamless integration for upscaling images by 4x — ideal for photographers, designers, and content creators.

[Live Demo](#) • [Portfolio](https://portflio-3.vercel.app/) • [GitHub](https://github.com/harshkushwaha7x)

</div>

---

<p align="center">
  <a href="https://github.com/harshkushwaha7x/Image-Upscaler-using-ESRGAN">
    <img src="https://img.shields.io/github/last-commit/harshkushwaha7x/Image-Upscaler-using-ESRGAN?style=flat-square" alt="last commit">
  </a>
  <a href="https://github.com/harshkushwaha7x/Image-Upscaler-using-ESRGAN">
    <img src="https://img.shields.io/github/languages/top/harshkushwaha7x/Image-Upscaler-using-ESRGAN?style=flat-square" alt="languages">
  </a>
  <a href="https://github.com/harshkushwaha7x/Image-Upscaler-using-ESRGAN/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="license" />
  </a>
  <img src="https://img.shields.io/badge/version-1.0.0-success?style=flat-square" alt="version" />
</p>

</div>

---

## 🧠 Overview
**ESRGAN Image Upscaler** is a production-ready AI-powered image enhancement platform that enables users to upscale images by 4x while maintaining high perceptual quality in real time.  
It combines **Enhanced Super-Resolution GAN**, **TensorFlow 2**, and **modern web technologies** to deliver seamless and scalable performance.  

Core highlights:
- 🎯 **4x Image Upscaling**: Transform low-resolution images to 4x their original size  
- 🤖 **AI-Powered Enhancement**: ESRGAN model for high-quality super-resolution  
- 💻 **Dual Interface**: CLI for batch processing and web UI for interactive use  
- ⚡ **GPU Acceleration**: Automatic GPU detection and utilization  
- 🌐 **Modern Web UI**: Flask-based interface with drag-and-drop support  

---

## 🚀 Key Features

### 🤖 AI-Powered Upscaling
- Enhanced Super-Resolution GAN (ESRGAN) for perceptual quality  
- TensorFlow Hub pre-trained model (~20MB)  
- Real-time image processing with progress tracking  
- Automatic model download on first run  

### 💻 Dual Interface
- Command-line tool for batch processing  
- Modern web interface with drag-and-drop upload  
- Side-by-side before/after comparison  
- Full-screen image modal view  

### 🖼️ Image Processing
- JPG and PNG format support  
- Alpha channel handling for transparent images  
- Automatic dimension validation (divisible by 4)  
- High-quality PNG output  

### ⚡ Performance
- GPU acceleration with CUDA support  
- Automatic CPU fallback  
- Efficient tensor operations  
- Processing statistics and timing  

---

## ⚙️ Tech Stack

### AI & Deep Learning
- TensorFlow 2.10+  
- TensorFlow Hub 0.12+  
- ESRGAN Model (TensorFlow 2 Edition)  
- OpenCV 4.8+ for image processing  

### Backend
- Python 3.7+  
- Flask 3.0.0  
- NumPy 1.24+ & Pillow 10.0+  
- Werkzeug 3.0.1  

### Frontend
- HTML5 + CSS3  
- Bootstrap 5  
- JavaScript (Vanilla)  
- Font Awesome Icons  

### DevOps
- Gunicorn 21.2.0+ (Production)  
- Docker-ready architecture  
- TensorFlow GPU support (CUDA)  

---

## 🧩 Architecture
```text
ESRGAN-Image-Upscaler/
├── test.py                # CLI script (177 lines)
│
├── data/                  # Data directories
│   ├── input/            # Place input images here
│   │   └── .gitkeep      # Maintains folder structure
│   └── output/           # Upscaled images saved here
│       └── .gitkeep      # Maintains folder structure
│
├── web/                  # Flask Web Application
│   ├── app.py           # Flask application (450 lines)
│   ├── start_server.py  # Server startup script (79 lines)
│   ├── static/          # Static assets
│   │   ├── uploads/     # Temporary uploaded files
│   │   │   └── .gitkeep
│   │   └── results/     # Temporary result files
│   │       └── .gitkeep
│   └── templates/       # HTML templates
│       ├── index.html   # Upload interface (650 lines)
│       └── result.html  # Results page (558 lines)
│
├── requirements.txt     # All dependencies (43 lines)
├── .gitignore          # Git ignore rules
├── LICENSE             # Apache 2.0 License
├── README.md           # This file
├── DOCUMENTATION.md    # Technical documentation (1137 lines)
└── PROJECT_REPORT.md   # Comprehensive project report
```

---

## 🧰 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.10+ (GPU support optional)
- OpenCV and NumPy
- 4GB RAM minimum (8GB+ recommended)
- NVIDIA GPU with CUDA support (optional, for faster processing)

### Installation
```bash
git clone https://github.com/harshkushwaha7x/Image-Upscaler-using-ESRGAN.git
cd Image-Upscaler-using-ESRGAN
```

#### Install Dependencies
```bash
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install tensorflow[and-cuda]
```

#### Verify GPU Support
```bash
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

#### Run Project

**Option 1: Command-Line Interface**
```bash
# Place images in data/input/
python test.py

# Results will be saved in data/output/
```

**Option 2: Web Interface**
```bash
# Navigate to web directory
cd web

# Start the server
python start_server.py

# Open http://localhost:5000 in your browser
```

---

## 🧠 AI Capabilities
- **ESRGAN Model**: Enhanced Super-Resolution Generative Adversarial Network  
- **4x Upscaling**: Increases image resolution by 400%  
- **Perceptual Quality**: Optimized for visual quality rather than pixel accuracy  
- **Automatic Model Download**: ~20MB model from TensorFlow Hub  
- **GPU Acceleration**: 10-15x faster processing with CUDA support  

---

## 🔌 API Endpoints

### Main Routes
- `GET /` - Main upload interface
- `POST /upscale` - Upload and process image
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /results/<filename>` - Serve result files

### System API
- `GET /api/status` - Check system status and GPU availability
- `GET /cleanup` - Clean up old temporary files (1-hour retention)

### Status Response Example
```json
{
  "status": "online",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_count": 1,
  "tensorflow_version": "2.10.0",
  "cuda_version": "/device:GPU:0",
  "timestamp": "2025-10-31T16:30:00"
}
```

---

## 📖 Detailed Usage

### Command-Line Interface

**Basic Usage:**
1. Place your images in `data/input/` directory
2. Run: `python test.py`
3. Find upscaled images in `data/output/` with `_rlt.png` suffix

**Example Output:**
```
======================================================================
ESRGAN Image Upscaler - TensorFlow 2 Edition
======================================================================

Loading ESRGAN model from TensorFlow Hub...
✓ Model loaded successfully!

Starting image processing...

1. Processing: my_photo
   ✓ Saved: my_photo_rlt.png (Time: 2.34s)

2. Processing: landscape
   ✓ Saved: landscape_rlt.png (Time: 3.12s)

======================================================================
Processing Complete!
======================================================================

Total images processed: 2
Average time per image: 2.73s
```

### Web Interface

**Features:**
- Drag-and-drop image upload
- Real-time image preview
- Processing with loading overlay
- Side-by-side before/after comparison
- Full-screen image modal
- Download enhanced images
- Processing statistics (time, resolution, file size)

**Supported Formats:**
- JPG, JPEG
- PNG (with alpha channel support)
- Maximum file size: 16MB

---

## ⚡ Performance

### Processing Speed

| Hardware | Image Size | Processing Time |
|----------|-----------|-----------------|
| CPU (Intel i7) | 512×512 | ~15-20 seconds |
| CPU (Intel i7) | 1024×1024 | ~60-80 seconds |
| GPU (RTX 3060) | 512×512 | ~2-3 seconds |
| GPU (RTX 3060) | 1024×1024 | ~8-12 seconds |

### Optimization Features
- GPU memory growth enabled (prevents OOM errors)
- Lazy model loading (loads on first use)
- Automatic file cleanup (1-hour retention)
- Efficient tensor operations
- Batch processing support (CLI)

---

## 🔧 Configuration

### GPU Acceleration
```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

### Environment Variables
```bash
# Optional configuration
export TFHUB_CACHE_DIR=/path/to/cache
export WEB_PORT=5000
export MAX_UPLOAD_SIZE=16777216  # 16MB
```

---

## ☁️ Deployment

### Development
```bash
cd web
python start_server.py
```

### Production (Gunicorn)
```bash
cd web
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 app:app
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "web.app:app"]
```

---

## 🔍 Troubleshooting

### Model Download Failed
```
Error: Failed to load model from TensorFlow Hub
```
**Solution:**
- Check your internet connection
- Model (~20MB) downloads automatically on first run
- Ensure sufficient disk space

### Out of Memory
```
Error: TensorFlow OOM (Out of Memory)
```
**Solution:**
- Reduce image size before processing
- Close other GPU-intensive applications
- Restart the application

### Slow Processing
**Solution:**
- Install TensorFlow with GPU support: `pip install tensorflow[and-cuda]`
- Check GPU availability with `/api/status` endpoint
- Update GPU drivers
- Use smaller images for faster processing

---

## 🤝 Contributing
1. Fork this repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add new feature"`)  
4. Push & open a Pull Request  

---

## 🪪 License
This project is licensed under the **Apache License 2.0** — see [LICENSE](https://github.com/harshkushwaha7x/Image-Upscaler-using-ESRGAN/blob/main/LICENSE).

---

## 🙏 Credits

- **Original ESRGAN Research:** [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
- **TensorFlow 2 Model:** [TensorFlow Hub - ESRGAN](https://tfhub.dev/captain-pool/esrgan-tf2/1)
- **Paper:** Wang, Xintao, et al. "ESRGAN: Enhanced super-resolution generative adversarial networks." ECCVW 2018.

---

## 📚 Citation

If you use this project in your research, please cite the original ESRGAN paper:

```bibtex
@InProceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
```

---

## 📬 Contact
**Harsh Kushwaha** — Developer & Maintainer  
- Portfolio: https://portflio-3.vercel.app/
- GitHub: https://github.com/harshkushwaha7x  
- LinkedIn: https://www.linkedin.com/in/harsh-kushwaha-7x
- Email: harshkushwaha4151@gmail.com

---

<div align="center">

**ESRGAN Image Upscaler** – Enhance Your Images with AI 🚀  
Built by <b>Harsh Kushwaha</b>
</div>
