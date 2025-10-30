# Railway Deployment Fix

## What Was Wrong

The deployment was failing because:

1. **Missing System Libraries**: OpenCV requires `libGL.so.1` which isn't available in Railway's environment
2. **Wrong OpenCV Package**: `opencv-python` includes GUI dependencies not needed for servers
3. **Healthcheck Timeout**: Railway's healthcheck was timing out before the app could start
4. **Model Loading**: The ESRGAN model (~20MB) takes time to download on first deployment
5. **Gunicorn Configuration**: The app wasn't configured properly for Gunicorn's preload mode

## What Was Fixed

### 1. Fixed OpenCV Dependencies ✨ CRITICAL FIX

**Problem:** 
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Solution:**

**a) Changed to headless OpenCV in `requirements.txt`:**
```python
# Before:
opencv-python>=4.8.0

# After:
opencv-python-headless>=4.8.0  # No GUI dependencies
```

**b) Created `nixpacks.toml`:**
```toml
[phases.setup]
nixPkgs = ["python310", "gcc", "libGL"]
```

This ensures Railway installs the required system libraries.

### 2. Updated `railway.json`

**Changes:**
- Changed healthcheck path from `/api/status` to `/` (simpler, faster)
- Added `healthcheckTimeout: 300` (5 minutes)
- Updated start command to use gunicorn config file

```json
{
  "deploy": {
    "startCommand": "cd web && gunicorn -c gunicorn_config.py app:app",
    "healthcheckPath": "/",
    "healthcheckTimeout": 300
  }
}
```

### 2. Created `web/gunicorn_config.py`

**Purpose:** Centralized Gunicorn configuration

**Key settings:**
- `preload_app = True` - Loads model before forking workers
- `timeout = 300` - 5 minutes for image processing
- `workers = 1` - Single worker (model is memory-intensive)
- Proper logging configuration

### 3. Updated `web/app.py`

**Changes:**
- Added `preload_model()` function
- Model loads when module is imported (for Gunicorn)
- Better error handling for model loading failures
- App starts even if model fails to load initially

```python
# Preload model when module is imported (for Gunicorn --preload)
try:
    preload_model()
except Exception as e:
    logger.warning(f"Initial model load failed: {e}")
```

### 4. Updated `Procfile`

**Change:**
```
web: cd web && gunicorn -c gunicorn_config.py app:app
```

## How to Deploy Now

### Step 1: Commit Changes

```bash
git add .
git commit -m "Fix Railway deployment configuration"
git push origin main
```

### Step 2: Railway Will Auto-Deploy

Railway detects the push and automatically redeploys.

**Expected timeline:**
- Build: ~2-3 minutes
- Model download: ~1-2 minutes (first time only)
- Healthcheck: ~30 seconds
- **Total: ~3-5 minutes**

### Step 3: Monitor Deployment

**In Railway Dashboard:**

1. Go to your project
2. Click "Deployments"
3. Watch the logs

**What to look for in logs:**

✅ **Good signs:**
```
Starting ESRGAN Image Upscaler...
Loading ESRGAN model from TensorFlow Hub...
Model loaded successfully!
Server is ready. Waiting for requests...
```

❌ **Warning signs (but OK):**
```
Model preload failed: ... Will load on first request.
```
This is OK - model will load on first request.

## Troubleshooting

### If Deployment Still Fails

#### Issue 1: Healthcheck Still Timing Out

**Solution:** Disable healthcheck temporarily

Update `railway.json`:
```json
{
  "deploy": {
    "startCommand": "cd web && gunicorn -c gunicorn_config.py app:app",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```
(Remove `healthcheckPath` and `healthcheckTimeout`)

#### Issue 2: Out of Memory

**Symptoms:**
```
Error: Killed (OOM)
```

**Solutions:**
1. Upgrade to Railway Pro (8GB RAM)
2. Or reduce max upload size in `app.py`:
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB instead of 16MB
   ```

#### Issue 3: Model Download Fails

**Symptoms:**
```
Error loading model: Failed to download from TensorFlow Hub
```

**Solutions:**
1. Wait and retry (temporary network issue)
2. Check Railway status page
3. Model will load on first request if preload fails

### Check Deploy Logs

**In Railway Dashboard:**

1. Go to Deployments
2. Click on the failed deployment
3. Click "Deploy Logs" tab

**Look for:**
- Python errors
- TensorFlow errors
- Memory errors
- Network errors

## Testing After Deployment

### 1. Check if App is Running

Visit your Railway URL: `https://your-app.up.railway.app`

You should see the upload interface.

### 2. Test Upload

1. Upload a small test image (512×512 or smaller)
2. Wait for processing (may take 30-60 seconds on first request)
3. Verify result downloads correctly

### 3. Check API Status

Visit: `https://your-app.up.railway.app/api/status`

Should return:
```json
{
  "status": "online",
  "model_loaded": true,
  ...
}
```

## Performance Notes

### First Request

- **First deployment**: Model downloads (~1-2 minutes)
- **First request**: Model loads into memory (~30-60 seconds)
- **Subsequent requests**: Fast (2-10 seconds depending on image size)

### Cold Starts (Free Tier)

Railway free tier may sleep your app after inactivity:
- **Sleep after**: 10 minutes of no requests
- **Wake up time**: 30-60 seconds
- **Solution**: Upgrade to Railway Pro ($5/month) for no sleep

## Success Checklist

- [ ] Code committed and pushed to GitHub
- [ ] Railway auto-deployed
- [ ] Build completed successfully
- [ ] Deploy logs show "Server is ready"
- [ ] App URL is accessible
- [ ] Homepage loads
- [ ] Image upload works
- [ ] Image processing completes
- [ ] Result can be downloaded

## Next Steps After Successful Deployment

1. **Test thoroughly** with various image sizes
2. **Monitor memory usage** in Railway dashboard
3. **Set up custom domain** (optional)
4. **Add environment variables** (optional):
   - `SECRET_KEY` for session security
5. **Upgrade to Pro** if you need:
   - More memory (up to 8GB)
   - No cold starts
   - Better performance

## Support

If you still have issues:

1. **Check Railway Status**: https://status.railway.app/
2. **Railway Discord**: https://discord.gg/railway
3. **Railway Docs**: https://docs.railway.app/
4. **Check Deploy Logs**: Look for specific error messages

---

**The fixes are now in place. Push to GitHub and Railway will automatically redeploy! 🚀**
