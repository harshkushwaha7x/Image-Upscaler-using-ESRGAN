#!/usr/bin/env python
"""
Startup script for ESRGAN Web Server
Provides a simple way to start the web interface
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    print("=" * 60)
    print("ESRGAN Image Upscaler - Web Interface")
    print("=" * 60)
    print()
    
    # Import and run the app
    from app import app, load_model, logger, WEB_PORT
    
    # Check model
    logger.info("Checking model availability...")
    logger.info("Model will be downloaded automatically from TensorFlow Hub on first run...")
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("✓ Model loaded successfully!")
    else:
        logger.error("✗ Failed to load model from TensorFlow Hub")
        logger.info("  Please check your internet connection and try again")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print()
    logger.info(f"Starting server on http://localhost:{WEB_PORT}")
    logger.info("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the app
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)
