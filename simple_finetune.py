#!/usr/bin/env python3
"""
Simplified fine-tuning wrapper that tries to work with or without TTS installed.
If TTS is not available, it delegates to AllTalk's finetune.py with proper error handling.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add AllTalk to path
alltalk_dir = Path(__file__).parent / "alltalk"
sys.path.insert(0, str(alltalk_dir))

DATASET_PATH = "./alltalk/finetune/input/nicholas_ofczarek"
OUTPUT_DIR = "./finetuned_models"
LANGUAGE = "de"
EPOCHS = 20

def main():
    print("Nicholas Ofczarek German TTS Fine-tuning")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Language: {LANGUAGE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Verify dataset exists
    dataset_dir = Path(DATASET_PATH)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1

    metadata_file = dataset_dir / "metadata.txt"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return 1

    with open(metadata_file, encoding='utf-8') as f:
        segments = f.readlines()

    print(f"\n[OK] Found {len(segments)} training segments")

    # Try importing TTS to check if it's available
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        print("[OK] TTS library is available")
        tts_available = True
    except ImportError as e:
        print(f"[INFO] TTS library not available: {e}")
        print("[INFO] Will attempt to run AllTalk finetune.py directly...")
        tts_available = False

    if not tts_available:
        # Try running finetune.py directly
        print("\nAttempting to run AllTalk finetune.py...")
        finetune_script = alltalk_dir / "finetune.py"

        if not finetune_script.exists():
            print(f"ERROR: finetune.py not found at {finetune_script}")
            return 1

        try:
            result = subprocess.run([
                sys.executable,
                str(finetune_script),
                "--dataset_path", DATASET_PATH,
                "--language", LANGUAGE,
                "--output_model_dir", OUTPUT_DIR,
                "--epochs", str(EPOCHS)
            ], cwd=str(alltalk_dir))

            if result.returncode == 0:
                print("\n" + "=" * 60)
                print("[OK] Fine-tuning completed successfully!")
                print("=" * 60)
                return 0
            else:
                print(f"\n[ERROR] finetune.py exited with code {result.returncode}")
                return result.returncode
        except Exception as e:
            print(f"[ERROR] Failed to run finetune.py: {e}")
            return 1

    # If we get here and TTS is available, run a simpler fine-tuning
    print("\n[TODO] Full TTS fine-tuning would run here")
    return 0

if __name__ == "__main__":
    sys.exit(main())
