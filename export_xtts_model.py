#!/usr/bin/env python3
"""
Convert a GPTTrainer training checkpoint (best_model.pth, keys prefixed with
`xtts.`) into an inference-ready XTTS model.pth (keys: mel_stats/gpt/hifigan_decoder).

This is the post-training export step. Without it, loading the trainer checkpoint
directly into an Xtts model silently fails to match the GPT weights and the model
collapses to near-silence at inference time.
"""
import argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, help="trainer checkpoint (best_model.pth)")
ap.add_argument("--out", required=True, help="output inference model.pth")
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

# Keep only the inference weights (xtts.*), strip the prefix. Drop training-only
# modules (dvae.*, torch_mel_spectrogram_*).
new_sd = {}
for k, v in sd.items():
    if k.startswith("xtts."):
        new_sd[k[len("xtts."):]] = v

print(f"[EXPORT] kept {len(new_sd)} / {len(sd)} weights")
torch.save({"model": new_sd}, args.out)  # XTTS loader expects a top-level 'model' key
print(f"[EXPORT] wrote {args.out}")
