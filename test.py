"""
ESRGAN Image Upscaler - Command Line Interface
===============================================

This script provides a command-line interface for upscaling images using the
Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) model.

The script processes all images in the 'data/input/' directory and saves the
upscaled results to 'data/output/' with a '_rlt.png' suffix.

Features:
    - Automatic model download from TensorFlow Hub
    - Batch processing of multiple images
    - GPU acceleration support (if available)
    - Progress tracking and timing statistics
    - Support for JPG and PNG formats
    - Automatic handling of alpha channels

Usage:
    1. Place images in 'data/input/' directory
    2. Run: python test.py
    3. Find upscaled images in 'data/output/' directory

"""

import os
import os.path as osp
import glob
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

# ============================================================================
# Configuration
# ============================================================================

# TensorFlow Hub model URL for ESRGAN
# This model provides 4x upscaling with enhanced perceptual quality
MODEL_URL = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# Input folder path - place images here for processing
# Supports wildcards for glob pattern matching
test_img_folder = 'data/input/*'

# Output folder path - upscaled images will be saved here
output_folder = 'data/output'

# ============================================================================
# Model Loading
# ============================================================================

# Enable download progress display for TensorFlow Hub
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

print('=' * 70)
print('ESRGAN Image Upscaler - TensorFlow 2 Edition')
print('=' * 70)
print('\nLoading ESRGAN model from TensorFlow Hub...')
print(f'Model URL: {MODEL_URL}')
print('Note: Model (~20MB) will be downloaded automatically on first run...\n')

# Load the pre-trained ESRGAN model from TensorFlow Hub
# The model is cached locally after first download
model = hub.load(MODEL_URL)

print('✓ Model loaded successfully!')
print('\nStarting image processing...\n')

# ============================================================================
# Image Processing Loop
# ============================================================================

# Initialize counters for statistics
idx = 0
total_time = 0

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(f'{idx}. Processing: {base}')
    
    # ========================================================================
    # Image Preprocessing
    # ========================================================================
    
    # Read image file and decode to tensor
    # TensorFlow automatically detects image format (JPG/PNG)
    img = tf.image.decode_image(tf.io.read_file(path))
    
    # Handle PNG images with alpha channel (transparency)
    # ESRGAN model expects RGB images, so we remove the alpha channel
    if img.shape[-1] == 4:
        img = img[..., :-1]  # Keep only RGB channels, discard alpha
    
    # Ensure image dimensions are divisible by 4
    # This is a requirement of the ESRGAN architecture
    hr_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) * 4
    img = tf.image.crop_to_bounding_box(img, 0, 0, hr_size[0], hr_size[1])
    
    # Convert image to float32 format (required by model)
    # Pixel values remain in [0, 255] range
    img = tf.cast(img, tf.float32)
    
    # Add batch dimension [height, width, channels] -> [1, height, width, channels]
    # Model expects batched input even for single images
    img = tf.expand_dims(img, 0)
    
    # ========================================================================
    # Super Resolution Inference
    # ========================================================================
    
    # Measure processing time for performance tracking
    start_time = time.time()
    
    # Perform super-resolution upscaling (4x)
    # Model outputs tensor with 4x dimensions
    output = model(img)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    total_time += elapsed
    
    # ========================================================================
    # Post-processing
    # ========================================================================
    
    # Remove batch dimension [1, height, width, channels] -> [height, width, channels]
    output = tf.squeeze(output)
    
    # Clip pixel values to valid range [0, 255]
    # Neural networks can sometimes produce values outside this range
    output = tf.clip_by_value(output, 0, 255)
    
    # Convert back to uint8 format and convert to NumPy array
    output = tf.cast(output, tf.uint8).numpy()
    
    # Convert from RGB (TensorFlow) to BGR (OpenCV) color space
    # OpenCV uses BGR format for image I/O operations
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # ========================================================================
    # Save Output Image
    # ========================================================================
    
    # Construct output file path with '_rlt' suffix
    output_path = os.path.join(output_folder, f'{base}_rlt.png')
    
    # Save upscaled image as PNG (lossless format)
    cv2.imwrite(output_path, output)
    
    print(f'   ✓ Saved: {base}_rlt.png (Time: {elapsed:.2f}s)\n')

# ============================================================================
# Summary Statistics
# ============================================================================

print('=' * 70)
print('Processing Complete!')
print('=' * 70)
print(f'\nTotal images processed: {idx}')

if idx > 0:
    print(f'Average time per image: {total_time/idx:.2f}s')
    print(f'Total processing time: {total_time:.2f}s')
else:
    print('\nNo images found in input folder.')
    print(f'Please place images in: {test_img_folder}')

print(f'\nUpscaled images saved to: {output_folder}/')
print('\n' + '=' * 70)
