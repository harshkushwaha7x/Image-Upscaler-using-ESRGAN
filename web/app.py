"""
ESRGAN Image Upscaler - Flask Web Application
==============================================

This module provides a web-based interface for the ESRGAN image upscaler.
It offers a modern, user-friendly UI for uploading images and viewing results.

Features:
    - Drag-and-drop image upload interface
    - Real-time image processing with ESRGAN model
    - Side-by-side before/after comparison
    - Download enhanced images
    - Processing statistics and system status API
    - Automatic cleanup of temporary files

Routes:
    GET  /           - Main upload interface
    POST /upscale    - Upload and process image
    GET  /uploads/<filename>  - Serve uploaded files
    GET  /results/<filename>  - Serve result files
    GET  /api/status - System status and GPU information
    GET  /cleanup    - Clean up old temporary files

Author: ESRGAN Team
License: Apache 2.0
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import time
from werkzeug.utils import secure_filename
from PIL import Image
import logging
from datetime import datetime

# ============================================================================
# Flask Application Configuration
# ============================================================================

app = Flask(__name__)
# Use environment variable for secret key, or generate a persistent one
# For production, set FLASK_SECRET_KEY environment variable
app.secret_key = os.environ.get('FLASK_SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# ============================================================================
# Application Constants
# ============================================================================

# TensorFlow Hub model URL for ESRGAN (4x upscaling)
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Web server configuration
WEB_PORT = 5000
SCALE_FACTOR = 4  # Image upscaling factor

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Directory Setup
# ============================================================================

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ============================================================================
# Environment Configuration
# ============================================================================

# Enable TensorFlow Hub download progress display
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# ============================================================================
# Global Model Instance
# ============================================================================

# Model is loaded lazily on first request to avoid startup delays
model = None

# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model():
    """
    Load the ESRGAN model from TensorFlow Hub.
    
    This function handles:
    - GPU detection and memory configuration
    - Model downloading and caching
    - Error handling and logging
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model
    if model is None:
        try:
            # Check GPU availability and configure memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPU available: {len(gpus)} device(s)")
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.info("No GPU found, using CPU")
            
            # Load pre-trained ESRGAN model from TensorFlow Hub
            # Model is cached locally after first download (~20MB)
            logger.info(f"Loading ESRGAN model from TensorFlow Hub...")
            logger.info(f"Model URL: {MODEL_URL}")
            model = hub.load(MODEL_URL)
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    return True

# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """
    Check if uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file_path):
    """
    Process an image using ESRGAN for 4x super-resolution upscaling.
    
    This function handles the complete image processing pipeline:
    1. Load and decode image from file
    2. Handle alpha channels (PNG transparency)
    3. Ensure dimensions are divisible by 4 (model requirement)
    4. Perform super-resolution inference
    5. Post-process and convert output
    
    Args:
        file_path (str): Path to the input image file
        
    Returns:
        tuple: (upscaled_image, processing_time) or (None, 0) on error
            - upscaled_image (numpy.ndarray): BGR format image ready for OpenCV
            - processing_time (float): Time taken for processing in seconds
    """
    try:
        # Ensure model is loaded
        if not load_model():
            return None, 0
        
        start_time = time.time()
        
        # ====================================================================
        # Image Preprocessing
        # ====================================================================
        
        # Read image file and decode to tensor (auto-detects format)
        img = tf.image.decode_image(tf.io.read_file(file_path))
        
        # Handle PNG images with alpha channel (transparency)
        # ESRGAN expects RGB images, so we discard the alpha channel
        if img.shape[-1] == 4:
            img = img[..., :-1]  # Keep only RGB channels
        
        # Ensure image dimensions are divisible by 4 (ESRGAN architecture requirement)
        hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
        
        # Validate minimum dimensions
        if hr_size[0] < 4 or hr_size[1] < 4:
            logger.error(f"Image dimensions too small: {hr_size[0]}x{hr_size[1]}")
            return None, 0
        
        img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
        
        # Convert to float32 format and add batch dimension
        # Model expects: [batch, height, width, channels]
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, 0)
        
        # ====================================================================
        # Super-Resolution Inference
        # ====================================================================
        
        # Perform 4x upscaling using ESRGAN model
        output = model(img)
        
        # ====================================================================
        # Post-Processing
        # ====================================================================
        
        # Remove batch dimension and clip values to valid range
        output = tf.squeeze(output)
        output = tf.clip_by_value(output, 0, 255)
        output = tf.cast(output, tf.uint8).numpy()
        
        # Convert from RGB (TensorFlow) to BGR (OpenCV) color space
        sr_img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        elapsed_time = time.time() - start_time
        
        return sr_img, elapsed_time
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, 0

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """
    Render the main upload interface.
    
    This route displays the homepage with drag-and-drop upload functionality.
    It also checks model availability and GPU status for display.
    
    Returns:
        str: Rendered HTML template
    """
    # Ensure model is loaded before displaying the page
    model_loaded = load_model()
    
    # Get GPU information for status display
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         gpu_available=gpu_available)

@app.route('/upscale', methods=['POST'])
def upscale():
    """
    Handle image upload and processing.
    
    This route:
    1. Validates uploaded file
    2. Saves file securely
    3. Processes image with ESRGAN
    4. Returns results page with before/after comparison
    
    Returns:
        str: Rendered result page or redirect on error
    """
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Additional backend validation for file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            flash('File size exceeds 16MB limit')
            return redirect(request.url)
        
        if file_size == 0:
            flash('File is empty')
            return redirect(request.url)
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image
        sr_img, processing_time = process_image(file_path)
        
        if sr_img is None:
            flash('Error processing image')
            return redirect(request.url)
        
        # Save the result with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(filename)[0]
        result_filename = f"sr_{base_name}_{timestamp}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # sr_img is already in BGR format from our processing
        cv2.imwrite(result_path, sr_img)
        
        # Get image dimensions for display
        lr_height, lr_width = cv2.imread(file_path).shape[:2]
        sr_height, sr_width = sr_img.shape[:2]
        
        # Calculate file sizes
        original_size = os.path.getsize(file_path)
        result_size = os.path.getsize(result_path)
        
        # Log the processing
        logger.info(f"Processed {filename}: {lr_width}x{lr_height} -> {sr_width}x{sr_height} in {processing_time:.2f}s")
        
        # Return result
        return render_template('result.html', 
                               original=f"uploads/{filename}",
                               result=f"results/{result_filename}",
                               processing_time=f"{processing_time:.2f}",
                               lr_size=f"{lr_width}x{lr_height}",
                               sr_size=f"{sr_width}x{sr_height}",
                               upscale_factor=SCALE_FACTOR,
                               original_size=original_size,
                               result_size=result_size)
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded image files.
    
    Args:
        filename (str): Name of the file to serve
        
    Returns:
        Response: File download response
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """
    Serve processed result image files.
    
    Args:
        filename (str): Name of the file to serve
        
    Returns:
        Response: File download response
    """
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/status')
def api_status():
    """
    API endpoint to check system status and capabilities.
    
    Returns JSON with:
    - Server status
    - Model loading status
    - GPU availability and count
    - TensorFlow version
    - CUDA device information
    - Current timestamp
    
    Returns:
        Response: JSON response with system information
    """
    model_loaded = model is not None
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    gpu_count = len(gpus)
    
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'gpu_available': gpu_available,
        'gpu_count': gpu_count,
        'tensorflow_version': tf.__version__,
        'cuda_version': tf.test.gpu_device_name() if gpu_available else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/cleanup')
def cleanup_files():
    """
    Clean up old temporary files from uploads and results folders.
    
    Removes files older than 1 hour to prevent disk space issues.
    This endpoint can be called manually or set up as a cron job.
    
    Returns:
        Response: JSON response with cleanup status and count
    """
    try:
        # Clean files older than 1 hour
        current_time = time.time()
        cleanup_time = 3600  # 1 hour
        
        cleaned_files = 0
        
        # Clean upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getmtime(file_path) > cleanup_time:
                    os.remove(file_path)
                    cleaned_files += 1
        
        # Clean results folder
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getmtime(file_path) > cleanup_time:
                    os.remove(file_path)
                    cleaned_files += 1
        
        return jsonify({
            'status': 'success',
            'cleaned_files': cleaned_files,
            'message': f'Cleaned {cleaned_files} old files'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == '__main__':
    # Pre-load model before starting server
    logger.info("Starting Enhanced ESRGAN Web Interface...")
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("Model loaded successfully!")
    else:
        logger.error("Failed to load model - application may not function correctly")
    
    # Start Flask development server
    # For production, use a WSGI server like Gunicorn
    logger.info(f"Server starting on http://0.0.0.0:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)