#!/usr/bin/env python3
"""
Rebuild a clean AllTalk metadata.txt by re-transcribing every WAV that
actually exists on disk in the dataset directory.

Fixes the desync where metadata.txt (169 rows) referenced files that no
longer existed (81 missing) while 169 on-disk WAVs were unreferenced.

Output format (AllTalk / XTTS): filename.wav|transcription|de
Rows with empty / too-short transcriptions are dropped.
"""

import os
import sys
import json
import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_dir(dataset_dir: str, language: str, model_size: str,
                   min_chars: int, min_words: int):
    dataset = Path(dataset_dir)
    wavs = sorted(dataset.glob("*.wav"))
    print(f"Found {len(wavs)} WAV files in {dataset}")
    if not wavs:
        sys.exit("No WAV files found.")

    # CPU int8 is the only option here (training GPU venv is separate);
    # medium gives solid German quality without being unbearably slow.
    print(f"Loading faster-whisper '{model_size}' (cpu/int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    rows = []
    dropped = []
    details = []
    for i, wav in enumerate(wavs, 1):
        segments, info = model.transcribe(str(wav), language=language)
        text = " ".join(s.text.strip() for s in segments).strip()
        # collapse whitespace
        text = " ".join(text.split())
        n_words = len(text.split())
        keep = len(text) >= min_chars and n_words >= min_words
        status = "KEEP" if keep else "DROP"
        print(f"[{i}/{len(wavs)}] {status} {wav.name}: {text[:70]}")
        rec = {"file": wav.name, "text": text, "chars": len(text),
               "words": n_words, "kept": keep}
        details.append(rec)
        if keep:
            rows.append(f"{wav.name}|{text}|{language}")
        else:
            dropped.append(wav.name)

    meta_path = dataset / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    # audit json alongside
    with open(dataset / "rebuild_audit.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Wrote {len(rows)} clean rows to {meta_path}")
    print(f"Dropped {len(dropped)} empty/short: {dropped}")
    print(f"Audit: {dataset / 'rebuild_audit.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"alltalk\finetune\input\nicholas_ofczarek")
    ap.add_argument("--language", default="de")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--min-words", type=int, default=3)
    args = ap.parse_args()
    transcribe_dir(args.dataset, args.language, args.model,
                   args.min_chars, args.min_words)
