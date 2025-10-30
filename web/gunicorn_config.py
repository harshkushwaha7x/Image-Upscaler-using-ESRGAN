"""
Gunicorn configuration for ESRGAN Image Upscaler
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Single worker due to model memory requirements
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutes for image processing
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'esrgan-upscaler'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload application for faster worker startup
preload_app = True

# Server hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting ESRGAN Image Upscaler...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready. Waiting for requests...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading workers...")
