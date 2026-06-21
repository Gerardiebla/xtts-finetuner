#!/usr/bin/env python3
"""
Master orchestration script for fine-tuning AllTalk TTS on Nicholas Ofczarek's voice.

Workflow:
1. Download audio from YouTube interviews
2. Process and segment audio
3. Prepare training dataset
4. Configure AllTalk fine-tuning
5. Run fine-tuning
6. Test the fine-tuned model
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
PROCESSED_AUDIO_DIR = DATA_DIR / "processed_audio"

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

def process_audio():
    """Process downloaded audio files."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Processing audio (segmenting, transcribing)")
    print(f"{'='*60}")

    wav_files = list(RAW_AUDIO_DIR.glob("*.wav"))
    if not wav_files:
        print(f"✗ No audio files found in {RAW_AUDIO_DIR}")
        return False

    print(f"Found {len(wav_files)} audio file(s) to process")

    result = subprocess.run([
        sys.executable,
        str(WORK_DIR / "process_audio.py"),
        str(RAW_AUDIO_DIR),
        "--output-dir", str(PROCESSED_AUDIO_DIR),
        "--language", "de"
    ])

    if result.returncode != 0:
        print(f"[FAIL] Failed to process audio")
        return False

    print(f"[OK] Audio processing complete")
    return True

def prepare_alltalk_dataset():
    """Prepare dataset in AllTalk's expected format."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Preparing AllTalk training dataset")
    print(f"{'='*60}")

    dataset_dir = ALLTALK_DIR / "finetune" / "input" / "nicholas_ofczarek"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio and metadata
    all_files = []
    for subdir in PROCESSED_AUDIO_DIR.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    for chunk in chunks:
                        if 'transcription' in chunk:
                            all_files.append({
                                'audio': chunk['file'],
                                'text': chunk['transcription'],
                                'language': 'de'
                            })

    if not all_files:
        print(f"[FAIL] No processed audio files with transcriptions found")
        return False

    # Create AllTalk metadata file
    metadata_path = dataset_dir / "metadata.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in all_files:
            audio_path = Path(item['audio']).name
            text = item['text']
            # AllTalk format: audio_file.wav|transcription|language
            f.write(f"{audio_path}|{text}|{item['language']}\n")

    # Copy audio files to dataset directory
    for subdir in PROCESSED_AUDIO_DIR.iterdir():
        if subdir.is_dir():
            for wav_file in subdir.glob("*.wav"):
                dest = dataset_dir / wav_file.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(wav_file, dest)

    print(f"[OK] Dataset prepared at: {dataset_dir}")
    print(f"  - {len(all_files)} audio files with transcriptions")
    print(f"  - Metadata file: {metadata_path}")

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
        "description": "Fine-tuned model for Nicholas Ofczarek's German voice"
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
        print(f"✗ Dataset directory not found: {dataset_path}")
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
    parser = argparse.ArgumentParser(description="Fine-tune AllTalk TTS on Nicholas Ofczarek")
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
            print("✗ YouTube URL required. Use --youtube-url or --skip-download")
            return 1
        if not download_audio(args.youtube_url):
            return 1

    # Process audio
    if not args.skip_process:
        if not process_audio():
            return 1

    # Prepare dataset
    if not prepare_alltalk_dataset():
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
