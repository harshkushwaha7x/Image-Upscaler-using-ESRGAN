# ESRGAN Image Upscaler - TensorFlow 2 Edition

🚀 **Enhanced Super-Resolution GAN for 4x Image Upscaling**

A production-ready TensorFlow 2 implementation of ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) for upscaling images by 4x with high perceptual quality.

## ✨ Features

- 🎯 **4x Image Upscaling** - Transform low-resolution images to 4x their original size
- 🤖 **Automatic Model Download** - Model downloads automatically from TensorFlow Hub (~20MB)
- ⚡ **GPU Acceleration** - Automatic GPU detection and utilization with TensorFlow
- 🌐 **Web Interface** - Modern Flask-based web UI with drag-and-drop support
- 💻 **Command Line** - Simple CLI for batch processing
- 📦 **Easy Installation** - Just install dependencies and run
- 🔧 **Production Ready** - Clean, documented, and deployment-ready code

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.10+
- OpenCV
- NumPy

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-Upscaler-Using-ESRGAN-Tensorflow.git
cd Image-Upscaler-Using-ESRGAN-Tensorflow

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Command Line Usage

```bash
# Place your images in data/input/
# Then run:
python test.py

# Results will be saved in data/output/
```

### Option 2: Web Interface

```bash
# Install all dependencies (if not already installed)
pip install -r requirements.txt

# Navigate to web directory
cd web

# Start the server
python start_server.py

# Open http://localhost:5000 in your browser
```

## 📖 Detailed Usage

### Command Line Interface

The `test.py` script processes all images in the `data/input/` folder:

```bash
python test.py
```

**Features:**
- Automatic model download from TensorFlow Hub on first run
- Processes all images in `data/input/`
- Saves results to `data/output/` with `_rlt.png` suffix
- Shows processing time for each image
- Supports JPG and PNG formats
- Handles images with alpha channels automatically

**Example Output:**
```
Loading ESRGAN TensorFlow 2 model from TensorFlow Hub...
Model loaded successfully!
Testing...

1. Processing: my_image
   Saved: my_image_rlt.png (Time: 2.34s)

Processing complete!
Total images processed: 1
Average time per image: 2.34s
```

### Web Interface

The web interface provides a user-friendly way to upscale images:

**Features:**
- Drag-and-drop image upload
- Real-time processing with progress indication
- Side-by-side before/after comparison
- Download enhanced images
- Processing statistics (time, resolution, file size)
- Automatic cleanup of old files

**API Endpoints:**

- `GET /` - Main upload interface
- `POST /upscale` - Upload and process image
- `GET /api/status` - Check system status and GPU availability
- `GET /cleanup` - Clean up old temporary files

**Example API Status Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_count": 1,
  "tensorflow_version": "2.10.0",
  "cuda_version": "/device:GPU:0"
}
```

## ⚙️ Configuration

### GPU Support

For faster processing with GPU acceleration:

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU availability
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

### Customization

Edit configuration in `test.py` or `web/app.py`:

```python
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"  # TensorFlow Hub model
SCALE_FACTOR = 4  # Upscaling factor
```

## 📁 Project Structure

```
Image-Upscaler-Using-ESRGAN-Tensorflow/
├── data/
│   ├── input/              # Place input images here
│   │   └── .gitkeep        # Maintains folder structure in Git
│   └── output/             # Upscaled images saved here
│       └── .gitkeep        # Maintains folder structure in Git
├── web/
│   ├── app.py              # Flask web application
│   ├── start_server.py     # Server startup script
│   ├── static/
│   │   ├── uploads/        # Temporary uploaded files
│   │   │   └── .gitkeep    # Maintains folder structure in Git
│   │   └── results/        # Temporary result files
│   │       └── .gitkeep    # Maintains folder structure in Git
│   └── templates/
│       ├── index.html      # Upload interface
│       └── result.html     # Results page
├── test.py                 # Command-line script
├── requirements.txt        # All dependencies (CLI + Web)
├── .gitignore              # Git ignore rules
├── LICENSE                 # Apache 2.0 License
└── README.md               # This file
```

## 🔧 Troubleshooting

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
- Check GPU availability with `/api/status` endpoint (web) or verify GPU detection
- Smaller images process faster

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

- **Original ESRGAN Research:** [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
- **TensorFlow 2 Model:** [TensorFlow Hub - ESRGAN](https://tfhub.dev/captain-pool/esrgan-tf2/1)
- **Paper:** Wang, Xintao, et al. "ESRGAN: Enhanced super-resolution generative adversarial networks." ECCVW 2018.

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

**Made with ❤️ using TensorFlow 2**
