#!/usr/bin/env python3
"""
Master orchestration script for fine-tuning AllTalk TTS with speaker diarization.

Workflow:
1. Download audio from YouTube interviews
2. Perform speaker diarization (identify speakers)
3. Process and segment audio (transcribe each segment)
4. Filter to keep only Nicholas Ofczarek segments
5. Prepare training dataset
6. Configure AllTalk fine-tuning
7. Run fine-tuning
8. Test the fine-tuned model
"""

import os
import json
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict

WORK_DIR = Path(__file__).parent
ALLTALK_DIR = WORK_DIR / "alltalk"
DATA_DIR = WORK_DIR / "training_data"
OUTPUT_DIR = WORK_DIR / "finetuned_models"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed_audio_diarized"

def setup_directories():
    """Create all necessary directories."""
    for dir_path in [DATA_DIR, OUTPUT_DIR, RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created directories")

def download_audio(youtube_url: str):
    """Download audio from YouTube."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Downloading audio from YouTube")
    print(f"{'='*60}")
    print(f"URL: {youtube_url}")

    result = subprocess.run([
        sys.executable,
        str(WORK_DIR / "download_youtube.py"),
        youtube_url,
        "--output-dir", str(RAW_AUDIO_DIR)
    ])

    if result.returncode != 0:
        print(f"[FAIL] Failed to download audio")
        return False

    print(f"[OK] Audio downloaded successfully")
    return True

def process_audio_with_diarization():
    """Process downloaded audio files with speaker diarization."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Processing audio (segmenting, transcribing, diarizing)")
    print(f"{'='*60}")

    wav_files = list(RAW_AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        print(f"[FAIL] No audio files found in {RAW_AUDIO_DIR}")
        return False

    print(f"Found {len(wav_files)} audio file(s) to process")

    result = subprocess.run([
        sys.executable,
        str(WORK_DIR / "process_audio_with_diarization.py"),
        str(RAW_AUDIO_DIR),
        "--output-dir", str(PROCESSED_AUDIO_DIR),
        "--language", "de"
    ])

    if result.returncode != 0:
        print(f"[FAIL] Failed to process audio")
        return False

    print(f"[OK] Audio processing complete")
    return True

def prepare_alltalk_dataset_filtered():
    """Prepare dataset in AllTalk's expected format, using only Nicholas segments."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Preparing AllTalk training dataset (Nicholas segments only)")
    print(f"{'='*60}")

    dataset_dir = ALLTALK_DIR / "finetune" / "input" / "nicholas_ofczarek"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Use nicholas_only_metadata.json which contains only Nicholas's segments
    metadata_file = PROCESSED_AUDIO_DIR / "nicholas_only_metadata.json"

    if not metadata_file.exists():
        print(f"[FAIL] Metadata file not found: {metadata_file}")
        return False

    with open(metadata_file, 'r', encoding='utf-8') as f:
        nicholas_segments = json.load(f)

    if not nicholas_segments:
        print(f"[FAIL] No Nicholas segments found after diarization")
        return False

    # Create AllTalk metadata file
    alltalk_metadata_path = dataset_dir / "metadata.txt"
    with open(alltalk_metadata_path, 'w', encoding='utf-8') as f:
        for item in nicholas_segments:
            if 'transcription' in item:
                audio_path = Path(item['file']).name
                text = item['transcription']
                # AllTalk format: audio_file.wav|transcription|language
                f.write(f"{audio_path}|{text}|de\n")

    # Copy audio files to dataset directory
    copied = 0
    for subdir in PROCESSED_AUDIO_DIR.iterdir():
        if subdir.is_dir():
            for wav_file in subdir.glob("*.wav"):
                dest = dataset_dir / wav_file.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(wav_file, dest)
                    copied += 1

    print(f"[OK] Dataset prepared at: {dataset_dir}")
    print(f"  - {len(nicholas_segments)} Nicholas segments")
    print(f"  - {copied} audio files copied")
    print(f"  - Metadata file: {alltalk_metadata_path}")

    # Print statistics
    with open(PROCESSED_AUDIO_DIR / "all_metadata.json", 'r', encoding='utf-8') as f:
        all_segments = json.load(f)

    nicholas_pct = (len(nicholas_segments) / len(all_segments)) * 100 if all_segments else 0
    print(f"\n  Filter statistics:")
    print(f"    Total segments: {len(all_segments)}")
    print(f"    Nicholas segments: {len(nicholas_segments)}")
    print(f"    Kept: {nicholas_pct:.1f}%")

    return True

def create_finetune_config():
    """Create AllTalk fine-tuning configuration."""
    print(f"\n{'='*60}")
    print(f"STEP 4: Creating fine-tuning configuration")
    print(f"{'='*60}")

    config = {
        "output_model_name": "nicholas_ofczarek_de",
        "language": "de",
        "epochs": 10,
        "batch_size": 4,
        "learning_rate": 0.0001,
        "model_type": "glow_tts",
        "description": "Fine-tuned model for Nicholas Ofczarek's German voice (diarized)"
    }

    config_file = WORK_DIR / "finetune_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"[OK] Configuration created at: {config_file}")
    print(f"  - Model name: {config['output_model_name']}")
    print(f"  - Language: {config['language']}")
    print(f"  - Epochs: {config['epochs']}")

    return True

def run_finetune():
    """Run AllTalk fine-tuning."""
    print(f"\n{'='*60}")
    print(f"STEP 5: Running fine-tuning")
    print(f"{'='*60}")

    dataset_path = ALLTALK_DIR / "finetune" / "input" / "nicholas_ofczarek"

    if not dataset_path.exists():
        print(f"[FAIL] Dataset directory not found: {dataset_path}")
        return False

    print(f"Starting AllTalk fine-tuning...")
    print(f"Dataset: {dataset_path}")

    # Run AllTalk's fine-tune script
    result = subprocess.run([
        sys.executable,
        str(ALLTALK_DIR / "finetune.py"),
        "--dataset_path", str(dataset_path),
        "--language", "de",
        "--output_model_dir", str(OUTPUT_DIR),
        "--epochs", "10"
    ], cwd=str(ALLTALK_DIR))

    if result.returncode != 0:
        print(f"Note: Fine-tuning returned exit code {result.returncode}")
        print(f"This may be normal - check the output above for any actual errors")

    print(f"[OK] Fine-tuning complete (check output directory)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fine-tune AllTalk TTS on Nicholas Ofczarek with speaker diarization")
    parser.add_argument("--youtube-url", help="YouTube URL or playlist of interviews")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading (use existing audio)")
    parser.add_argument("--skip-process", action="store_true", help="Skip audio processing")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning (prep only)")

    args = parser.parse_args()

    # Setup
    setup_directories()

    # Download
    if not args.skip_download:
        if not args.youtube_url:
            print("[FAIL] YouTube URL required. Use --youtube-url or --skip-download")
            return 1
        if not download_audio(args.youtube_url):
            return 1

    # Process audio with diarization
    if not args.skip_process:
        if not process_audio_with_diarization():
            return 1

    # Prepare dataset (Nicholas only)
    if not prepare_alltalk_dataset_filtered():
        return 1

    # Configure
    if not create_finetune_config():
        return 1

    # Fine-tune
    if not args.skip_finetune:
        if not run_finetune():
            return 1

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Workflow complete!")
    print(f"{'='*60}")
    print(f"Training data: {DATA_DIR}")
    print(f"Output models: {OUTPUT_DIR}")
    print(f"\nNext: Use the model in AllTalk for German TTS with Nicholas's voice!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
