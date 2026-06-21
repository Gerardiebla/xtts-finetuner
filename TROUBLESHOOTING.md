# Troubleshooting Guide

## Installation Issues

### FFmpeg Not Found

**Error:**
```
[FAIL] FFmpeg not found
```

**Solution:**

1. **Windows - Download Manually:**
   - Visit: https://ffmpeg.org/download.html
   - Download latest Windows build (ffmpeg-full)
   - Extract to: `C:\ffmpeg\`
   - Add to PATH:
     - Right-click Start → System
     - Advanced system settings → Environment Variables
     - New: `FFMPEG_PATH = C:\ffmpeg\bin`

2. **Windows - Chocolatey:**
   ```powershell
   choco install ffmpeg
   ```

3. **Windows - Windows Package Manager:**
   ```powershell
   winget install FFmpeg
   ```

4. **Verify Installation:**
   ```powershell
   ffmpeg -version
   ```

---

### Python/Environment Issues

**Error: "Python not found"**
```powershell
# Activate virtual environment
cd "C:\Users\gack8\OneDrive\Documents\Claude code projects\TTS cloning"
.\venv\Scripts\Activate.ps1
```

**Error: "module not found"**
```powershell
# Install missing package
.\venv\Scripts\python.exe -m pip install <package-name>
```

**Error: "Permission denied"**
```powershell
# Run PowerShell as Administrator
# Then try again
```

---

### GPU/CUDA Issues

**Error: "CUDA not available (CPU mode)"**

The system will still work, but training will be slow. Options:

1. **Check NVIDIA drivers:**
   ```powershell
   nvidia-smi
   ```

2. **Ensure PyTorch CUDA version matches your drivers**

3. **For now, training will use CPU** - it's just slower

**Error: "Out of memory"**

Reduce batch size in `finetune_config.json`:
```json
{
  "batch_size": 1,
  "epochs": 10
}
```

---

## Runtime Issues

### Audio Download Problems

**Error: "youtube-dl error" / "yt-dlp error"**

```powershell
# Update yt-dlp
.\venv\Scripts\python.exe -m pip install --upgrade yt-dlp

# Try again
.\venv\Scripts\python.exe download_youtube.py "URL"
```

**Error: "Video unavailable" or "Age restricted"**

- Video may be private, age-restricted, or deleted
- Try a different interview
- Ensure URL is correct
- Check video is publicly accessible

**Error: "No audio found"**

- Video may be audio-only or have issues
- Try: `.\venv\Scripts\python.exe download_youtube.py "URL" --output-dir ./raw_audio`
- Check `raw_audio/` folder for downloaded files

---

### Audio Processing Issues

**Error: "No audio files found"**

```
✗ No audio files found in ./training_data/raw_audio
```

**Solution:**
1. Check files were downloaded: `ls ./training_data/raw_audio/`
2. Supported formats: `.wav`, `.mp3`, `.m4a`
3. Run download first: `python download_youtube.py "URL"`

**Error: "Whisper transcription failed"**

Often happens with:
- Poor audio quality
- Heavy background noise
- Non-German speech
- Very long segments

**Solutions:**
- Use higher quality interviews
- Reduce segment duration in `process_audio.py`:
  ```python
  max_duration: float = 15  # Instead of 30
  ```
- Manually review and fix transcriptions

**Error: "Out of memory during processing"**

- Reduce segment maximum length
- Process fewer files at once
- Edit `process_audio.py`:
  ```python
  max_duration: float = 15  # was 30
  min_duration: float = 3   # was 5
  ```

---

### Training Issues

**Error: "Dataset directory not found"**

```
✗ Dataset directory not found: .../finetune/input/nicholas_ofczarek
```

**Solution:**
1. Ensure audio was processed: `python process_audio.py ./raw_audio`
2. Verify metadata files exist: `ls ./training_data/processed_audio/*/metadata.json`
3. Run setup again: `python finetune_setup.py --skip-download --skip-process`

**Error: "CUDA out of memory during training"**

```
RuntimeError: CUDA out of memory
```

**Solution:**

1. **Reduce batch size** in `finetune_config.json`:
   ```json
   {
     "batch_size": 1
   }
   ```

2. **Reduce epochs** (train for fewer iterations):
   ```json
   {
     "epochs": 5
   }
   ```

3. **Close other applications** to free GPU memory

4. **Use CPU instead** (slow but works):
   - Edit `finetune_setup.py` to add `--device cpu`

**Error: "Training loss is NaN or infinite"**

Indicates a learning rate problem:

1. **Reduce learning rate** in `finetune_config.json`:
   ```json
   {
     "learning_rate": 0.00001
   }
   ```

2. **Reduce batch size:**
   ```json
   {
     "batch_size": 2
   }
   ```

3. **Use different audio** - current data may be problematic

**Error: "Training is very slow"**

If training is slower than expected:

1. **Check GPU is being used:**
   - Look for `cuda:0` in logs
   - Run: `nvidia-smi` in another terminal
   - If GPU usage is 0%, training is on CPU

2. **If on CPU:** That's normal, training is slow
   - Estimated: 30-60 min per hour of audio
   - Leave it running overnight

3. **If on GPU but still slow:**
   - GTX 1650 is relatively slow
   - Expected: 2-4 hours per hour of audio

---

### Testing Issues

**Error: "Model not found after training"**

Check output directory:
```powershell
ls ./finetuned_models/
ls ./alltalk/finetune/output/
```

**Error: "Generated audio quality is poor"**

- Model may need more training data
- Train with more interviews (longer duration)
- Increase epochs
- Use better quality audio source

---

## Common Problems & Solutions

### Problem: "Whisper transcription is inaccurate"

**Why:** Whisper works best with clear audio and consistent speech

**Solutions:**
1. Download interviews with better audio quality
2. Manually fix transcriptions in `metadata.json`
3. Use shorter audio segments (reduce `max_duration`)

### Problem: "Training takes forever"

**Expected times (GTX 1650):**
- 30 min audio → 1-2 hours
- 1 hour audio → 2-3 hours
- 3 hours audio → 6-10 hours

**It's normal!** Leave it running.

To speed up:
- Reduce epochs
- Reduce audio length
- Close other applications

### Problem: "Generated voice doesn't sound like Nicholas"

**Why:** Insufficient training data or poor quality

**Solutions:**
1. **Use longer interviews** (aim for 3+ hours total)
2. **Use multiple interviews** (different topics, tones)
3. **Train longer** (increase epochs to 20-30)
4. **Better audio quality** (clear, minimal background noise)

### Problem: "Pitch or tone is incorrect"

- Use interviews with natural speech patterns
- Avoid dramatic readings or performances
- Include variety of sentences and tones
- Train with more data

---

## Debug Mode

### Enable Detailed Logging

```powershell
# Set debug environment variable
$env:DEBUG = "1"

# Run with verbose output
.\venv\Scripts\python.exe finetune_setup.py --youtube-url "URL" 2>&1 | Tee-Object debug.log
```

### Inspect Intermediate Files

```powershell
# Check downloaded audio
dir ./training_data/raw_audio/

# Check processed segments
dir ./training_data/processed_audio/*/

# Check transcriptions
cat ./training_data/processed_audio/*/metadata.json

# Check dataset
dir ./alltalk/finetune/input/nicholas_ofczarek/
cat ./alltalk/finetune/input/nicholas_ofczarek/metadata.txt
```

### Manual Testing

```powershell
# Test audio download
.\venv\Scripts\python.exe download_youtube.py "URL" --output-dir test_audio

# Test audio processing
.\venv\Scripts\python.exe process_audio.py test_audio --language de

# Test with single file
.\venv\Scripts\python.exe process_audio.py test_audio --language de --max-duration 10
```

---

## Getting Help

### Check Existing Logs

```powershell
# AllTalk training logs
cat ./alltalk/logs/
cat ./finetuned_models/logs/

# Process logs
cat ./training_data/processed_audio/*/metadata.json
```

### Review Configuration

```powershell
# Check current config
cat ./finetune_config.json

# Check YouTube download config
cat ./download_youtube.py | grep -A 10 "ydl_opts"
```

### Test Step by Step

```powershell
# Just download (don't process)
.\venv\Scripts\python.exe download_youtube.py "URL"

# Just process (don't train)
.\venv\Scripts\python.exe finetune_setup.py --youtube-url "URL" --skip-finetune

# Just train (skip download/process)
.\venv\Scripts\python.exe finetune_setup.py --skip-download --skip-process
```

---

## Still Stuck?

1. **Check README.md** - Comprehensive reference
2. **Check QUICKSTART.md** - Quick examples
3. **Review log files** - Often shows the actual error
4. **Try simpler audio** - Shorter, clearer segments
5. **Reduce complexity** - Smaller batch size, fewer epochs
6. **Leave it running** - Training takes time!

---

## Environment Information

When reporting issues, include:

```powershell
# Python version
python --version

# CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Installed packages
pip list | grep -E "torch|whisper|yt-dlp"

# System info
wmic os get caption, version
nvidia-smi
```

Good luck with your fine-tuning! 🚀
