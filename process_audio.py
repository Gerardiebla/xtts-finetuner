#!/usr/bin/env python3
"""
Process raw audio files for TTS fine-tuning:
- Convert to 22050 Hz WAV format
- Segment into 5-30 second chunks
- Transcribe using Whisper
- Create metadata for training
"""

import os
import json
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
import wave
import contextlib

def install_dependencies():
    """Install required packages."""
    packages = ["librosa", "scipy", "openai-whisper"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def convert_audio_format(input_file: str, output_file: str, sample_rate: int = 22050):
    """Convert audio to WAV format with specified sample rate using ffmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-i", input_file, "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-y", output_file
        ], check=True, capture_output=True)
        print(f"Converted: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e}")
        return False

def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        import librosa
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0

def segment_audio(audio_file: str, output_dir: str, min_duration: float = 5, max_duration: float = 30):
    """Segment audio into chunks of min_duration to max_duration seconds."""
    import librosa
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    chunks = []
    start_time = 0
    chunk_idx = 0

    while start_time < duration:
        end_time = min(start_time + max_duration, duration)
        chunk_duration = end_time - start_time

        if chunk_duration >= min_duration:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = y[start_sample:end_sample]

            chunk_file = output_dir / f"chunk_{chunk_idx:04d}.wav"
            sf.write(chunk_file, chunk_audio, sr)
            chunks.append({
                'file': str(chunk_file),
                'start_time': start_time,
                'end_time': end_time,
                'duration': chunk_duration,
            })
            print(f"Created chunk {chunk_idx}: {chunk_duration:.2f}s")
            chunk_idx += 1

        start_time = end_time

    return chunks

def transcribe_audio(audio_file: str, language: str = "de") -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
    except ImportError:
        print("Installing whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        import whisper

    print(f"Transcribing: {audio_file}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, language=language)

    transcription = result["text"].strip()
    print(f"Transcription: {transcription}")
    return transcription

def process_directory(input_dir: str, output_dir: str = "./processed_data", language: str = "de"):
    """Process all audio files in a directory."""
    install_dependencies()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.m4a"))

    all_metadata = []

    for audio_file in audio_files:
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file.name}")
        print(f"{'='*60}")

        # Convert to WAV if needed
        if audio_file.suffix.lower() != ".wav":
            temp_wav = output_dir / f"{audio_file.stem}_temp.wav"
            if not convert_audio_format(str(audio_file), str(temp_wav)):
                continue
            audio_to_process = temp_wav
        else:
            audio_to_process = audio_file

        # Create subdirectory for this file's segments
        file_output_dir = output_dir / audio_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Segment audio
        chunks = segment_audio(str(audio_to_process), str(file_output_dir))

        # Transcribe each chunk
        for chunk in chunks:
            try:
                transcription = transcribe_audio(chunk['file'], language)
                chunk['transcription'] = transcription
                all_metadata.append(chunk)
            except Exception as e:
                print(f"Error transcribing {chunk['file']}: {e}")

        # Save metadata
        metadata_file = file_output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to: {metadata_file}")

    # Save overall metadata
    overall_metadata = output_dir / "all_metadata.json"
    with open(overall_metadata, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete! Total chunks: {len(all_metadata)}")
    print(f"Metadata saved to: {overall_metadata}")

    return all_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio for TTS fine-tuning")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output-dir", default="./processed_data", help="Output directory")
    parser.add_argument("--language", default="de", help="Language code (de for German)")
    parser.add_argument("--min-duration", type=float, default=5, help="Minimum chunk duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30, help="Maximum chunk duration in seconds")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.language)
