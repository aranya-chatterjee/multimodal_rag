# Streamlit Cloud Setup Guide

## If you get "installer returned a non-zero exit code" error:

### Option 1: Use the updated requirements.txt (already pushed)
The requirements.txt has been updated with pinned versions that are more compatible with Streamlit Cloud.

### Option 2: If PyTorch/Whisper fails to install
If video processing packages (torch, whisper) are causing issues, you can:

1. **Temporarily disable video features** by using `requirements-lite.txt`:
   - Rename `requirements-lite.txt` to `requirements.txt` in your repo
   - This version only supports document processing (no video)

2. **Or install packages in stages** - Streamlit Cloud may timeout with large packages

### Option 3: Check Streamlit Cloud logs
1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Check the "Logs" tab for specific error messages
4. Common issues:
   - **Memory limit**: PyTorch is very large (~2GB)
   - **Timeout**: Installation takes too long
   - **Missing system dependencies**: ffmpeg for video processing

### Option 4: Use Streamlit Community Cloud limits
Streamlit Community Cloud has:
- 1GB RAM limit
- 1 CPU core
- Installation timeout limits

**Solutions:**
- Use CPU-only PyTorch (already in requirements.txt)
- Consider using smaller Whisper models
- Process videos locally and upload transcripts instead

### Recommended: Make video processing optional
You can modify the app to gracefully handle missing video dependencies:

```python
# In video_loader.py, the app already checks for availability
# If video processing fails, document processing will still work
```

## Current Status
- ✅ Requirements.txt updated with compatible versions
- ✅ Added .streamlit/config.toml for proper configuration
- ✅ All dependencies pinned to specific versions

## Next Steps
1. **Restart your Streamlit Cloud app** after the requirements.txt update
2. **Check the logs** if it still fails
3. **Consider using requirements-lite.txt** if video processing isn't critical

## Alternative: Deploy without video processing
If you only need document processing:
1. Use `requirements-lite.txt` (rename to `requirements.txt`)
2. The app will work for documents but video features will be disabled
