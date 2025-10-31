#!/usr/bin/env python
"""
ESRGAN Web Server - Startup Script
===================================

This script provides a convenient way to start the ESRGAN web interface.
It handles model loading verification and provides user-friendly startup messages.

Usage:
    python start_server.py
    
Or from the web directory:
    cd web
    python start_server.py

The server will start on http://localhost:5000 by default.

Author: ESRGAN Team
License: Apache 2.0
"""

import os
import sys

# ============================================================================
# Path Configuration
# ============================================================================

# Add parent directory to Python path to allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Display startup banner
    print("="*60)
    print("ESRGAN Image Upscaler - Web Interface")
    print("="*60)
    print()
    
    # Import Flask app and dependencies
    # Importing here to ensure path is set up first
    from app import app, load_model, logger, WEB_PORT
    
    # ========================================================================
    # Model Loading and Verification
    # ========================================================================
    
    logger.info("Checking model availability...")
    logger.info("Model will be downloaded automatically from TensorFlow Hub on first run...")
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("✓ Model loaded successfully!")
    else:
        # Model failed to load - give user option to continue
        logger.error("✗ Failed to load model from TensorFlow Hub")
        logger.error("  Please check your internet connection and try again")
        logger.error("  The application will not function without the model")
        print()
        response = input("Continue anyway? (not recommended) (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Exiting...")
            sys.exit(1)
        logger.warning("Starting server without model - image processing will fail")
    
    # ========================================================================
    # Start Web Server
    # ========================================================================
    
    print()
    logger.info(f"Starting server on http://localhost:{WEB_PORT}")
    logger.info("Press Ctrl+C to stop the server")
    print("="*60)
    print()
    
    # Run the Flask application
    # Using threaded=True for better concurrent request handling
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)
