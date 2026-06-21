# XTTS Studio

A lightweight, local GUI pipeline for fine-tuning [XTTS v2](https://github.com/coqui-ai/TTS/blob/main/TTS/tts/models/xtts) (multilingual voice synthesis) to clone German speakers. Runs entirely on Windows with a GTX 1650 4GB GPU.

**Three steps, one window:**
1. **Prep** — add cleaned single-speaker audio clips → auto-trim silence → segment to ≤11s → German transcription → `metadata.txt`
2. **Train** — fine-tune XTTS on your dataset (auto-exports inference-ready model)
3. **Test** — generate German speech in your cloned voice

## Setup

### Requirements
- Windows 10/11 with Python 3.11+
- NVIDIA GPU (tested on GTX 1650 4GB; CPU training is possible but very slow)
- CUDA 12.4 (optional; CPU-only mode works)
- ~3–4 GB free disk space for the base XTTS v2 model

### Installation

1. **Clone this repo:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/xtts-studio.git
   cd xtts-studio
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv_tts
   venv_tts\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install coqui-tts torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install transformers==4.57.1 faster-whisper librosa soundfile numpy
   ```

   **Note:** If the torch wheels are large and hit Windows Defender locks, download them locally first:
   ```bash
   pip download torch torchaudio -d wheels --index-url https://download.pytorch.org/whl/cu124
   pip install --no-index --find-links wheels torch torchaudio
   ```

4. **Set the Coqui license flag:**
   ```bash
   # Windows PowerShell
   $env:COQUI_TOS_AGREED = "1"
   ```
   Or add it to your shell profile permanently.

### Usage

Launch the GUI:
```bash
venv_tts\Scripts\python.exe audio_prep_gui.py
```

#### Tab 1: Prep
- **Add files** — select cleaned, single-speaker audio clips (wav, mp3, flac, m4a, ogg)
  - ⚠️ **Must be one speaker only** — no diarisation step here; the tool assumes clean input
  - ⚠️ **Minimal background noise** — the tool trims silence but does NOT denoise
- **Silence threshold (top_db)** — lower = more aggressive trimming (default 30)
- **Min/Max clip duration** — enforces ≤11s for XTTS compatibility (default 2–11s)
- **Transcribe** — tick to auto-transcribe with faster-whisper (German) and generate `metadata.txt`
- **Process** → outputs trimmed `clip_XXXX.wav` files and `metadata.txt`

#### Tab 2: Train
- **Dataset folder** — path to the folder from Prep (must contain `metadata.txt` and clips)
- **Output folder** — where the fine-tuned model is saved (auto-exports `model.pth` on completion)
- **Hyperparameters:**
  - **Epochs** — default 12 (lower than the initial run; best-model checkpoint sidesteps overfit)
  - **Learning rate** — default 1e-5 (much higher than initial 5e-6; enables better learning in fewer epochs)
  - **Batch size** — default 2 (fits 4GB VRAM via system-RAM spillover on Windows)
  - **Grad accum** — default 4 (gradient accumulation for effective larger batches)
- **Start training** → runs the fine-tune as a subprocess with live log streaming
- **Stop** — terminates training if needed
- ✅ When done, the Test tab auto-points to your fresh model

#### Tab 3: Test
- **Model folder** — where your trained `model.pth` + `config.json` + `vocab.json` live
- **Reference clip** — a short (≤10s) audio sample of the speaker, used for voice conditioning (default: first clip from prep)
- **Text** — German text to synthesize
- **Language** — default `de`; can use other XTTS-supported languages
- **Temperature** — 0.7 (default); lower = more predictable, higher = more varied
- **Generate** → synthesizes audio and auto-plays it (model is cached for fast repeats)

## What's Inside

- **`audio_prep_gui.py`** — the three-tab Tk GUI (Prep, Train, Test)
- **`train_xtts_local.py`** — standalone trainer; uses GPTTrainer from coqui-tts
  - Loads base XTTS v2 model via ModelManager + HuggingFace fallback
  - Trains on your dataset with efficient low-VRAM settings
  - **Auto-exports inference model** at end (strips trainer-checkpoint prefix, wraps in correct format)
  - Outputs checkpoints + TensorBoard logs to `output/training/`
- **`export_xtts_model.py`** — standalone export utility (trainer checkpoint → inference model)
- **`test_inference.py`** — standalone voice-cloning test script
- **`requirements_tts.txt`** — frozen pip dependencies (ref only; install via steps above)

## Quirks & Notes

### The ±11s Clip Limit
XTTS v2's fine-tune dataset loader has a hard limit on clip duration (~11s). Longer clips are silently rejected via infinite recursion in `__getitem__`, causing a `RecursionError`. The Prep tab handles this by segmenting on silence + hard-capping at 11s.

### Checkpoint → Inference Model
The trainer outputs `best_model.pth`, which is NOT directly usable by the `Xtts` inference loader. The keys are prefixed `xtts.` and include optimizer/scheduler state. **The trainer auto-exports `model.pth`** (correct format) at the end. If needed manually, use `export_xtts_model.py`.

### Learning Rate & Epochs
The initial 20-epoch run at LR 5e-6 barely moved the model (4.03 → 3.82 eval loss, ~5%). We recommend:
- **LR 1e-5** — 4× higher, better learning per epoch
- **~12 epochs** — lower than 20; the model overfits after the best-model checkpoint anyway
- **Use best_model.pth** — frozen at lowest eval loss, before the overfit tail

### Noise & Speaker Contamination
Remaining audio quality issues are data-side, not model-side:
- **Noise** — the Prep tool trims silence but does NOT denoise. Feed it clean, low-noise source audio.
- **Speaker swapping** — if your source interviews have both the speaker and an interviewer, and the transcription didn't separate them, both voices end up in the training set. Use single-speaker clips only, or manually edit `metadata.txt` post-Prep.

### GPU Memory
On a 4GB GTX 1650:
- **Training** uses ~3–3.5 GB with batch_size=2, grad_accum=4. Windows spills excess to system RAM (slow but stable).
- **Inference** (Test tab) uses ~1–2 GB. Safe even if training is running (though not recommended).

### CPU-Only Mode
CUDA is optional. Set `torch.cuda.is_available()` to `False` in the relevant scripts, or:
```bash
pip install torch torchaudio  # CPU-only wheels
```
Training will be much slower (hours → days for 12 epochs).

## Architecture

### XTTS v2 Overview
- **Base model**: 1B-parameter multilingual GPT-based TTS
- **Fine-tuning**: GPTTrainer (coqui-tts) trains only the GPT decoder on new speakers' voice + text
- **Voice conditioning**: speaker embeddings extracted from a reference clip via HiFi-GAN
- **Output**: 24 kHz waveform

### Why Coqui-TTS?
The original `TTS==0.22.0` package requires MSVC on Windows (no prebuilt wheel for the Cython monotonic-alignment extension). The **coqui-tts** fork (0.27.5) ships prebuilt cp311 wheels and works out-of-the-box on Windows.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ImportError: cannot import name 'isin_mps_friendly'` | transformers >= 5.12 removed it | `pip install transformers==4.57.1` |
| `RecursionError` during training prep | clips > 11s in the dataset | Re-run Prep tab; verify metadata.txt clips exist |
| WinError 32 on torch wheel install | Windows Defender scanning large file | Download wheels locally, use `--no-index` |
| CUDA out of memory | batch size too high | Lower `--batch-size` (try 1) |
| Model generates only silence | Loading a trainer checkpoint instead of exported model | Use `model.pth` (exported), not `best_model.pth` (trainer state) |

## Contributing & Licensing

This is a minimal wrapper around [coqui-tts](https://github.com/coqui-ai/TTS) and the excellent [faster-whisper](https://github.com/guillaumekln/faster-whisper). It's shared in the spirit of open research.

**Coqui TTS** is CPML-licensed; agree to the terms before use (set `COQUI_TOS_AGREED=1`).

## Citation & Credits

- **XTTS v2**: [Gal et al., 2024](https://arxiv.org/abs/2305.13620)
- **Coqui TTS**: [The Coqui Team](https://github.com/coqui-ai/TTS)
- **faster-whisper**: [Guillaume Klein](https://github.com/guillaumekln/faster-whisper)
- **GUI**: Built with Python Tkinter

---

**Status:** Beta. Tested on Windows 11 + Python 3.11 + GTX 1650 4GB.

For questions or issues, open a GitHub issue.
