#!/usr/bin/env python3
"""
Verify that the TTS fine-tuning environment is set up correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [FAIL] Python 3.8+ required")
        return False
    return True

def check_venv():
    """Check if running in virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"[OK] Virtual environment active")
        return True
    else:
        print("[FAIL] Not running in virtual environment")
        print("  Run: .\\venv\\Scripts\\Activate.ps1")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace("-", "_")

    try:
        __import__(import_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} not found")
        return False

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  [OK] CUDA available - {device_name} ({device_count} device(s))")
            return True
        else:
            print(f"  [WARN] CUDA not available (CPU mode)")
            return True
    except ImportError:
        print(f"[FAIL] PyTorch not found")
        return False

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version
            version_line = result.stdout.split('\n')[0]
            print(f"[OK] FFmpeg: {version_line.split(' ')[0:3]}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[FAIL] FFmpeg not found")
        print("  Download from: https://ffmpeg.org/download.html")
        print("  Or: choco install ffmpeg")
        return False

def check_directories():
    """Check if required directories exist."""
    work_dir = Path(__file__).parent
    alltalk_dir = work_dir / "alltalk"

    if alltalk_dir.exists():
        print(f"[OK] AllTalk repository found")
        return True
    else:
        print(f"[FAIL] AllTalk repository not found at {alltalk_dir}")
        return False

def check_packages():
    """Check essential Python packages."""
    packages = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]

    all_ok = True
    for package, import_name in packages:
        if not check_package(package, import_name):
            all_ok = False

    return all_ok

def main():
    print("=" * 60)
    print("TTS Fine-tuning Setup Verification")
    print("=" * 60 + "\n")

    checks = [
        ("Python Version", check_python),
        ("PyTorch & CUDA", check_pytorch),
        ("Core Python Packages", check_packages),
        ("FFmpeg", check_ffmpeg),
        ("AllTalk Repository", check_directories),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Error: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("[SUCCESS] All checks passed! Ready to fine-tune.")
        print("\nNext steps:")
        print("1. Find a YouTube video with Nicholas Ofczarek speaking German")
        print("2. Run: python finetune_setup.py --youtube-url <URL>")
        print("3. Wait for training to complete")
        return 0
    else:
        print("[ALERT] Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install FFmpeg from https://ffmpeg.org/download.html")
        print("- Ensure you're in the virtual environment: .\\venv\\Scripts\\Activate.ps1")
        print("- Install missing packages: python -m pip install <package>")
        return 1

if __name__ == "__main__":
    sys.exit(main())
