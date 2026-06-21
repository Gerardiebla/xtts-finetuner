#!/usr/bin/env python3
"""Quick inference test using the fine-tuned XTTS checkpoint."""
import argparse
from pathlib import Path

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

THIS_DIR = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint-dir", default=str(
    THIS_DIR / "finetuned_models" / "nicholas_ofczarek_de" / "training" /
    "XTTS_FT-June-20-2026_09+44PM-0000000"))
ap.add_argument("--model-file", default="best_model.pth")
ap.add_argument("--speaker-wav", default=str(
    THIS_DIR / "alltalk" / "finetune" / "input" / "nicholas_clips" / "v1_utt_0000.wav"))
ap.add_argument("--text", default=(
    "Guten Tag, ich freue mich sehr, heute mit Ihnen über dieses Thema "
    "zu sprechen. Es ist mir eine große Ehre, hier zu sein."))
ap.add_argument("--out", default=str(THIS_DIR / "test_output.wav"))
args = ap.parse_args()

ckpt_dir = Path(args.checkpoint_dir)
config = XttsConfig()
config.load_json(str(ckpt_dir / "config.json"))

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=str(ckpt_dir / args.model_file),
    vocab_path=str(THIS_DIR / "xtts_base" / "vocab.json"),
    use_deepspeed=False,
)
if torch.cuda.is_available():
    model.cuda()

print("[INFER] Computing speaker latents from reference clip...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[args.speaker_wav]
)

print("[INFER] Generating speech...")
out = model.inference(
    text=args.text,
    language="de",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.7,
    repetition_penalty=5.0,
    length_penalty=1.0,
    enable_text_splitting=True,
)

import numpy as np
wav = np.asarray(out["wav"])
print(f"[INFER] wav len={len(wav)} ({len(wav)/24000:.2f}s) max_amp={np.abs(wav).max():.4f} rms={np.sqrt((wav**2).mean()):.4f}")

import soundfile as sf
sf.write(args.out, wav, 24000)
print(f"[DONE] Wrote {args.out}")
