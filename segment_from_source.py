#!/usr/bin/env python3
"""
Re-segment the full-length source interviews into short, sentence-aligned
clips suitable for XTTS v2 fine-tuning (<= ~11s each, with matching text).

The previous pipeline cut flat 30s chunks, which XTTS rejects (it needs
<= ~11s). This uses faster-whisper word timestamps to cut at natural
sentence / pause boundaries, resamples to 22050 Hz mono, and writes a
clean metadata.txt (filename|text|de).
"""

import os
import glob
import argparse
from pathlib import Path

import librosa
import soundfile as sf
from faster_whisper import WhisperModel

MAX_DUR = 11.0     # hard upper bound (XTTS-friendly)
MIN_DUR = 2.0      # drop anything shorter
SOFT_DUR = 7.0     # prefer to break at sentence end once past this
SENT_END = (".", "?", "!", "…")


def group_words(words):
    """Group whisper word objects into utterances <= MAX_DUR, breaking at
    sentence boundaries when possible."""
    utts, cur, start = [], [], None
    for w in words:
        if start is None:
            start = w.start
        cur.append(w)
        dur = w.end - start
        text = "".join(x.word for x in cur).strip()
        ends_sentence = text.endswith(SENT_END)
        if (dur >= SOFT_DUR and ends_sentence) or dur >= MAX_DUR:
            utts.append((start, w.end, text))
            cur, start = [], None
    if cur:
        text = "".join(x.word for x in cur).strip()
        utts.append((start, cur[-1].end, text))
    return utts


def process(src_paths, out_dir, model_size, language):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading faster-whisper '{model_size}' (cpu/int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    rows = []
    idx = 0
    for si, src in enumerate(src_paths):
        tag = f"v{si+1}"
        print(f"\n=== {os.path.basename(src)} -> prefix {tag}_ ===")
        # full audio at 22050 mono for slicing
        audio, sr = librosa.load(src, sr=22050, mono=True)

        segments, _ = model.transcribe(
            src, language=language, word_timestamps=True,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
        )
        words = []
        for seg in segments:
            if seg.words:
                words.extend(seg.words)
        print(f"  {len(words)} words; grouping into utterances...")

        for (start, end, text) in group_words(words):
            dur = end - start
            if dur < MIN_DUR or not text or len(text) < 3:
                continue
            a = int(start * sr)
            b = int(end * sr)
            clip = audio[a:b]
            if len(clip) < int(MIN_DUR * sr):
                continue
            fname = f"{tag}_utt_{idx:04d}.wav"
            sf.write(out / fname, clip, sr)
            rows.append(f"{fname}|{text}|{language}")
            idx += 1

        print(f"  running total clips: {idx}")

    meta = out / "metadata.txt"
    with open(meta, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nWrote {len(rows)} clips + {meta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=r"training_data\raw_audio")
    ap.add_argument("--out", default=r"alltalk\finetune\input\nicholas_clips")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--language", default="de")
    args = ap.parse_args()

    srcs = sorted(glob.glob(os.path.join(args.src_dir, "*.wav")))
    if not srcs:
        raise SystemExit(f"No source wavs in {args.src_dir}")
    print("Sources:", [os.path.basename(s) for s in srcs])
    process(srcs, args.out, args.model, args.language)
