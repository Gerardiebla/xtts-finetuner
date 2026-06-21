#!/usr/bin/env python3
"""
Download audio from YouTube videos for TTS fine-tuning.
Uses yt-dlp to download and ffmpeg to convert to WAV format.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def install_ytdlp():
    """Install yt-dlp if not already installed."""
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])

def download_youtube_audio(url, output_dir="./raw_audio"):
    """Download audio from YouTube URL and convert to WAV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        install_ytdlp()
        import yt_dlp

    # Download audio as mp3
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }

    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base_name = Path(filename).stem
        wav_file = output_dir / f"{base_name}.wav"
        print(f"Successfully downloaded: {wav_file}")
        return str(wav_file)

def download_playlist(playlist_url, output_dir="./raw_audio"):
    """Download all videos from a YouTube playlist."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        install_ytdlp()
        import yt_dlp

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / '%(playlist_title)s_%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }

    print(f"Downloading playlist from: {playlist_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

    wav_files = list(output_dir.glob("*.wav"))
    print(f"Downloaded {len(wav_files)} audio files")
    return wav_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio from YouTube for TTS fine-tuning")
    parser.add_argument("url", help="YouTube URL or playlist URL")
    parser.add_argument("--output-dir", default="./raw_audio", help="Output directory for audio files")
    parser.add_argument("--playlist", action="store_true", help="Download entire playlist")

    args = parser.parse_args()

    if args.playlist:
        download_playlist(args.url, args.output_dir)
    else:
        download_youtube_audio(args.url, args.output_dir)
