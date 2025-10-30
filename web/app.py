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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Configuration
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
WEB_PORT = 5000
SCALE_FACTOR = 4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Set environment variable for download progress
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Load the model
model = None

def load_model():
    global model
    if model is None:
        try:
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"GPU available: {len(gpus)} device(s)")
                # Enable memory growth to avoid OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.info("No GPU found, using CPU")
            
            # Load model from TensorFlow Hub
            logger.info(f"Loading ESRGAN model from TensorFlow Hub...")
            logger.info(f"Model URL: {MODEL_URL}")
            model = hub.load(MODEL_URL)
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    return True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file_path):
    """Process an image and return the super-resolution result using TensorFlow 2"""
    try:
        # Ensure model is loaded
        if not load_model():
            return None, 0
        
        start_time = time.time()
        
        # Read and preprocess image
        img = tf.image.decode_image(tf.io.read_file(file_path))
        
        # If PNG with alpha channel, remove it
        if img.shape[-1] == 4:
            img = img[..., :-1]
        
        # Ensure dimensions are divisible by 4 (model requirement)
        hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
        img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
        
        # Convert to float32 and add batch dimension
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, 0)
        
        # Perform super resolution
        output = model(img)
        
        # Postprocess
        output = tf.squeeze(output)
        output = tf.clip_by_value(output, 0, 255)
        output = tf.cast(output, tf.uint8).numpy()
        
        # Convert RGB to BGR for OpenCV
        sr_img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        elapsed_time = time.time() - start_time
        
        return sr_img, elapsed_time
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, 0

@app.route('/')
def index():
    # Check if model is loaded
    model_loaded = load_model()
    
    # Get system info for display
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         gpu_available=gpu_available)

@app.route('/upscale', methods=['POST'])
def upscale():
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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/status')
def api_status():
    """API endpoint to check system status"""
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
    """Clean up old uploaded and result files"""
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

logger.info("Initializing ESRGAN Web Interface...")
load_model()

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    
    # Use PORT environment variable for deployment platforms (Render, Heroku, etc.)
    # Fallback to WEB_PORT (5000) for local development
    port = int(os.environ.get('PORT', WEB_PORT))
    logger.info(f"Server starting on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)