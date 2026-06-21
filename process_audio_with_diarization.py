#!/usr/bin/env python3
"""
Advanced audio processing with speaker diarization:
- Convert to 22050 Hz WAV format
- Segment into 5-30 second chunks
- Perform speaker diarization (identify who is speaking)
- Transcribe using Whisper
- Filter segments by speaker (keep Nicholas, remove interviewer)
- Create metadata for training
"""

import os
import json
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

def install_dependencies():
    """Install required packages."""
    packages = ["librosa", "scipy", "openai-whisper", "pyannote.audio"]
    for package in packages:
        try:
            if package == "openai-whisper":
                import whisper
            elif package == "pyannote.audio":
                from pyannote.audio import Pipeline
            else:
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
        ], check=True, capture_output=True, timeout=300)
        print(f"  Converted: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error converting {input_file}: {e}")
        return False

def perform_speaker_diarization(audio_file: str) -> Dict:
    """
    Perform speaker diarization to identify speakers in the audio.
    Returns a dict with speaker timings.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        print("  [WARN] pyannote.audio not available, skipping diarization")
        return {}

    try:
        print(f"  Running speaker diarization...")

        # Use HuggingFace pipeline for speaker diarization
        # This requires a HuggingFace token in ~/.cache/huggingface/
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        # Run diarization
        diarization = pipeline(audio_file)

        # Extract speaker segments
        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append({
                'start': turn.start,
                'end': turn.end
            })

        print(f"  [OK] Identified {len(speakers)} speakers")
        return {
            'diarization': diarization,
            'speakers': speakers
        }

    except Exception as e:
        print(f"  [WARN] Diarization failed: {e}")
        print(f"  Continuing without speaker filtering...")
        return {}

def segment_audio(audio_file: str, output_dir: str, diarization_info: Dict = None,
                  min_duration: float = 5, max_duration: float = 30):
    """
    Segment audio into chunks with speaker information.
    If diarization_info provided, identify speakers in each chunk.
    """
    import librosa
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading audio file: {audio_file}")
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

            chunk_info = {
                'file': str(chunk_file),
                'start_time': start_time,
                'end_time': end_time,
                'duration': chunk_duration,
            }

            # Identify speaker for this chunk if diarization available
            if diarization_info and 'diarization' in diarization_info:
                try:
                    diarization = diarization_info['diarization']
                    chunk_mid = (start_time + end_time) / 2

                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if turn.start <= chunk_mid <= turn.end:
                            chunk_info['speaker'] = speaker
                            chunk_info['speaker_confidence'] = 'high'
                            break

                    if 'speaker' not in chunk_info:
                        chunk_info['speaker'] = 'unknown'
                        chunk_info['speaker_confidence'] = 'low'
                except Exception as e:
                    print(f"    [WARN] Error identifying speaker: {e}")
                    chunk_info['speaker'] = 'unknown'

            chunks.append(chunk_info)
            print(f"  Created chunk {chunk_idx}: {chunk_duration:.2f}s", end="")
            if 'speaker' in chunk_info:
                print(f" [{chunk_info['speaker']}]")
            else:
                print()
            chunk_idx += 1

        start_time = end_time

    return chunks

def transcribe_audio(audio_file: str, language: str = "de") -> str:
    """Transcribe audio using OpenAI Whisper."""
    try:
        import whisper
    except ImportError:
        print("  Installing whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        import whisper

    print(f"  Transcribing: {Path(audio_file).name}")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, language=language)

    transcription = result["text"].strip()
    return transcription

def identify_nicholas(speakers_dict: Dict) -> str:
    """
    Identify which speaker label is Nicholas by analyzing characteristics.
    In a real scenario, you might need manual input or speaker embeddings.
    For now, we'll use heuristics or ask the user.
    """
    if not speakers_dict:
        return None

    print(f"\n  Found speakers: {list(speakers_dict.keys())}")
    print(f"  Note: Speaker labels are IDs assigned by the diarization model")
    print(f"  Typically, the speaker with more speaking time is the main interviewee")

    # Calculate speaking time per speaker
    speaker_durations = {}
    for speaker, segments in speakers_dict.items():
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)
        speaker_durations[speaker] = total_duration

    # Assume the speaker with most talking time is Nicholas (interviewee)
    nicholas_speaker = max(speaker_durations, key=speaker_durations.get)

    print(f"\n  Likely Nicholas speaker (most speaking time): {nicholas_speaker}")
    print(f"  Speaking time: {speaker_durations[nicholas_speaker]:.1f}s")

    return nicholas_speaker

def process_directory(input_dir: str, output_dir: str = "./processed_data",
                     language: str = "de", nicholas_speaker: str = None):
    """Process all audio files in a directory with speaker diarization."""
    install_dependencies()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")) + list(input_dir.glob("*.m4a"))

    all_metadata = []
    nicholas_segments = []

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

        # Perform speaker diarization
        print(f"\nPerforming speaker diarization...")
        diarization_info = perform_speaker_diarization(str(audio_to_process))

        # Identify Nicholas if we have speakers
        local_nicholas = nicholas_speaker
        if diarization_info and 'speakers' in diarization_info and not nicholas_speaker:
            local_nicholas = identify_nicholas(diarization_info['speakers'])

        # Create subdirectory for this file's segments
        file_output_dir = output_dir / audio_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Segment audio
        print(f"\nSegmenting audio...")
        chunks = segment_audio(str(audio_to_process), str(file_output_dir), diarization_info)

        # Transcribe each chunk
        print(f"\nTranscribing segments...")
        for chunk in chunks:
            try:
                transcription = transcribe_audio(chunk['file'], language)
                chunk['transcription'] = transcription

                # Track Nicholas segments
                if local_nicholas and chunk.get('speaker') == local_nicholas:
                    nicholas_segments.append(chunk)
                    chunk['keep'] = True
                elif not diarization_info.get('diarization'):
                    # No diarization, keep all segments
                    nicholas_segments.append(chunk)
                    chunk['keep'] = True
                else:
                    chunk['keep'] = False

                all_metadata.append(chunk)

                keep_status = "[KEEP]" if chunk.get('keep') else "[FILTER]"
                print(f"  {keep_status} {Path(chunk['file']).name}: {transcription[:50]}...")

            except Exception as e:
                print(f"  [SKIP] Error transcribing {chunk['file']}: {e}")

        # Save metadata
        metadata_file = file_output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved metadata to: {metadata_file}")

    # Save overall metadata
    overall_metadata = output_dir / "all_metadata.json"
    with open(overall_metadata, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    # Save filtered metadata (Nicholas only)
    nicholas_metadata = output_dir / "nicholas_only_metadata.json"
    with open(nicholas_metadata, 'w', encoding='utf-8') as f:
        json.dump(nicholas_segments, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Processing complete!")
    print(f"{'='*60}")
    print(f"Total segments processed: {len(all_metadata)}")
    print(f"Nicholas segments: {len(nicholas_segments)}")
    if len(all_metadata) > 0:
        nicholas_pct = (len(nicholas_segments) / len(all_metadata)) * 100
        print(f"Percentage Nicholas: {nicholas_pct:.1f}%")
    print(f"\nMetadata files:")
    print(f"  - All segments: {overall_metadata}")
    print(f"  - Nicholas only: {nicholas_metadata}")

    return nicholas_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio with speaker diarization")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output-dir", default="./processed_data", help="Output directory")
    parser.add_argument("--language", default="de", help="Language code (de for German)")
    parser.add_argument("--min-duration", type=float, default=5, help="Minimum chunk duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30, help="Maximum chunk duration in seconds")
    parser.add_argument("--nicholas-speaker", help="Specify which speaker ID is Nicholas")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.language, args.nicholas_speaker)
