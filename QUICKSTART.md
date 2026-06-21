# Quick Start Guide - TTS Fine-tuning

## Setup (One-time)

### 1. Verify Setup
```powershell
cd "C:\Users\gack8\OneDrive\Documents\Claude code projects\TTS cloning"
.\venv\Scripts\python.exe verify_setup.py
```

This checks:
- ✓ Python version
- ✓ Virtual environment
- ✓ PyTorch & CUDA
- ✓ FFmpeg
- ✓ Required packages

### 2. Install System Dependencies (if needed)

**FFmpeg** (required):
- Download: https://ffmpeg.org/download.html
- Or via Chocolatey: `choco install ffmpeg`
- Verify: `ffmpeg -version`

**C++ Build Tools** (for full AllTalk support - optional for now):
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "C++ Build Tools" workload

## Usage

### Step 1: Find YouTube Interview(s)

Search for interviews with Nicholas Ofczarek speaking German:
- Siemens convention talks
- Tech conferences
- Podcast appearances
- Company events

Example: https://www.youtube.com/watch?v=VIDEOID

### Step 2: Run the Fine-tuning Pipeline

```powershell
cd "C:\Users\gack8\OneDrive\Documents\Claude code projects\TTS cloning"

# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Run the complete workflow
.\venv\Scripts\python.exe finetune_setup.py --youtube-url "https://www.youtube.com/watch?v=VIDEOID"
```

The script will automatically:
1. ✓ Download audio from YouTube
2. ✓ Process & segment into 5-30 second chunks
3. ✓ Transcribe each segment using Whisper (German)
4. ✓ Prepare dataset for AllTalk
5. ✓ Run fine-tuning (10 epochs by default)
6. ✓ Save trained model

**Expected Time:** 
- 1 hour audio → 2-3 hours training
- 2 hours audio → 4-6 hours training
- (On GTX 1650; GPU times vary)

### Step 3: Test the Model

After training completes, test your fine-tuned model:

```powershell
cd alltalk
.\..\..\venv\Scripts\python.exe -c "
# Load and test the model
import torch
# (See testing section in README.md)
"
```

## Troubleshooting

### Issue: "youtube not found"
**Solution:** Install yt-dlp
```powershell
.\venv\Scripts\python.exe -m pip install yt-dlp
```

### Issue: "ffmpeg not found"
**Solution:** Install FFmpeg
- Windows: Download from https://ffmpeg.org/download.html
- Add to PATH or install via: `choco install ffmpeg`

### Issue: "Low VRAM" or Out of Memory
**Solution:** Reduce batch size in `finetune_config.json`:
```json
{
  "batch_size": 2,
  "epochs": 10
}
```

### Issue: Slow Training
**Solution:** 
- Check GPU is being used: Check training logs for "cuda:0" mentions
- Reduce audio quality or length: Edit `process_audio.py`
- Add more GPU VRAM if possible

### Issue: Poor Audio Quality in Output
**Solution:**
- Download higher quality interviews
- Use longer training (increase epochs)
- Use interviews with consistent sound quality
- Avoid background noise

## Advanced Usage

### Download Multiple Interviews

```powershell
# Download from playlist
.\venv\Scripts\python.exe finetune_setup.py --youtube-url "https://www.youtube.com/playlist?list=..." --youtube-url

# Or run separately
.\venv\Scripts\python.exe download_youtube.py "URL1" --output-dir .\training_data\raw_audio
.\venv\Scripts\python.exe download_youtube.py "URL2" --output-dir .\training_data\raw_audio

# Then process all together
.\venv\Scripts\python.exe process_audio.py .\training_data\raw_audio --language de
```

### Adjust Training Parameters

Edit `finetune_config.json`:
```json
{
  "output_model_name": "nicholas_ofczarek_de",
  "language": "de",
  "epochs": 20,           # More iterations = better quality but slower
  "batch_size": 4,        # Larger = faster but uses more VRAM
  "learning_rate": 0.0001 # Lower = more stable but slower
}
```

### Process Existing Audio

If you already have audio files:

```powershell
# Place audio in: ./training_data/raw_audio/
# Then:
.\venv\Scripts\python.exe finetune_setup.py --skip-download
```

### Prepare Data Only (No Training)

```powershell
# Just prepare the dataset, don't train yet
.\venv\Scripts\python.exe finetune_setup.py --youtube-url "..." --skip-finetune
```

## File Locations

| Directory | Purpose |
|-----------|---------|
| `training_data/raw_audio/` | Downloaded YouTube audio |
| `training_data/processed_audio/` | Segmented & transcribed audio |
| `finetuned_models/` | Output: trained models |
| `alltalk/finetune/input/nicholas_ofczarek/` | AllTalk training dataset |

## What Gets Created

After successful training:
- `finetuned_models/nicholas_ofczarek_de/` - The trained model
- `training_data/processed_audio/*/metadata.json` - Transcriptions
- `finetuned_models/logs/` - Training logs

## Next Steps

1. Integrate the model into AllTalk web interface
2. Test TTS output quality
3. Fine-tune hyperparameters for better results
4. Deploy to your application

## Support

For issues:
1. Check `verify_setup.py` output
2. Read `README.md` for detailed information
3. Check AllTalk documentation: https://github.com/erew123/alltalk_tts

---

**Ready to start?**

```powershell
cd "C:\Users\gack8\OneDrive\Documents\Claude code projects\TTS cloning"
.\venv\Scripts\python.exe verify_setup.py
```
