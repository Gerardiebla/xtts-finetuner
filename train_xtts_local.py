#!/usr/bin/env python3
"""
Standalone, headless XTTS v2 GPT fine-tuner.

Reuses the proven AllTalk `train_gpt` recipe (GPTTrainer / GPTArgs /
GPTTrainerConfig) but:
  * runs without the gradio UI,
  * skips the Whisper step (we already have transcripts),
  * downloads the XTTS v2 base checkpoint via coqui ModelManager,
  * is tuned for a low-VRAM (4 GB GTX 1650) Windows machine.

Pipeline:
  1. Read dataset metadata.txt  (filename|text|de)
  2. Split into metadata_train.csv / metadata_eval.csv  (audio_file|text|speaker_name)
  3. Download XTTS v2 base files if missing
  4. Fine-tune for N epochs
"""

import os
import sys
import csv
import glob
import random
import argparse
from pathlib import Path

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs, GPTTrainer, GPTTrainerConfig,
)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

THIS_DIR = Path(__file__).resolve().parent
BASE_FILES = ["model.pth", "config.json", "vocab.json", "mel_stats.pth", "dvae.pth"]


# --------------------------------------------------------------------------- #
# Step 1 + 2: build train/eval CSVs from our metadata.txt
# --------------------------------------------------------------------------- #
def prepare_csvs(dataset_dir: Path, speaker: str, eval_pct: float, seed: int):
    meta = dataset_dir / "metadata.txt"
    if not meta.exists():
        sys.exit(f"metadata.txt not found in {dataset_dir}")

    rows = []
    with open(meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            fname, text = parts[0].strip(), parts[1].strip()
            if not text or not (dataset_dir / fname).exists():
                continue
            rows.append((fname, text))

    if not rows:
        sys.exit("No usable rows found in metadata.txt")

    random.Random(seed).shuffle(rows)
    n_eval = max(1, int(len(rows) * eval_pct))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    train_csv = dataset_dir / "metadata_train.csv"
    eval_csv = dataset_dir / "metadata_eval.csv"
    for path, subset in ((train_csv, train_rows), (eval_csv, eval_rows)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|")
            w.writerow(["audio_file", "text", "speaker_name"])
            for fname, text in subset:
                w.writerow([fname, text, speaker])

    print(f"[PREP] {len(train_rows)} train / {len(eval_rows)} eval rows")
    print(f"[PREP] {train_csv}")
    print(f"[PREP] {eval_csv}")
    return str(train_csv), str(eval_csv)


# --------------------------------------------------------------------------- #
# Step 3: download XTTS v2 base checkpoint
# --------------------------------------------------------------------------- #
def ensure_base_model(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    if all((base_dir / f).exists() for f in BASE_FILES):
        print(f"[BASE] All base files present in {base_dir}")
        return base_dir

    print("[BASE] Downloading XTTS v2 base model via ModelManager ...")
    os.environ.setdefault("COQUI_TOS_AGREED", "1")  # bypass interactive CPML license prompt
    mm = ModelManager()
    model_path, _, _ = mm.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    src = Path(model_path)
    print(f"[BASE] Downloaded to {src}; copying required files -> {base_dir}")

    import shutil
    for f in BASE_FILES:
        cand = src / f
        if cand.exists():
            shutil.copy2(cand, base_dir / f)
        else:
            print(f"[BASE] WARNING: {f} not in download dir, will try HF fallback")

    # Fallback: fetch any missing files straight from the HF repo.
    missing = [f for f in BASE_FILES if not (base_dir / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download
        for f in missing:
            print(f"[BASE] HF fallback download: {f}")
            p = hf_hub_download(repo_id="coqui/XTTS-v2", filename=f)
            shutil.copy2(p, base_dir / f)

    missing = [f for f in BASE_FILES if not (base_dir / f).exists()]
    if missing:
        sys.exit(f"[BASE] Could not obtain base files: {missing}")
    print(f"[BASE] Base model ready in {base_dir}")
    return base_dir


# --------------------------------------------------------------------------- #
# Step 4: train (adapted from AllTalk train_gpt)
# --------------------------------------------------------------------------- #
def train(args, train_csv, eval_csv, base_dir: Path):
    out_path = Path(args.output) / "training"
    out_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dev = torch.cuda.current_device()
        free_gb = (torch.cuda.get_device_properties(dev).total_memory
                   - torch.cuda.memory_allocated(dev)) / (1024 ** 3)
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(dev)}  VRAM ~{free_gb:.2f} GB")
        if free_gb < 12:
            print("[TRAIN] NOTE: <12GB VRAM. On Windows, CUDA can spill into system RAM "
                  "(slow but avoids hard OOM). Keep batch_size small.")
    else:
        print("[TRAIN] WARNING: CUDA not available — training on CPU will be extremely slow.")

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(os.path.abspath(train_csv)),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=args.language,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=args.max_wav_length,
        max_text_length=200,
        mel_norm_file=str(base_dir / "mel_stats.pth"),
        dvae_checkpoint=str(base_dir / "dvae.pth"),
        xtts_checkpoint=str(base_dir / "model.pth"),
        tokenizer_file=str(base_dir / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        epochs=args.epochs,
        output_path=str(out_path),
        model_args=model_args,
        run_name="XTTS_FT",
        project_name="XTTS_trainer",
        run_description="GPT XTTS German fine-tune (Nicholas Ofczarek)",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=args.workers,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=args.save_step,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18],
                             "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"[TRAIN] Loaded {len(train_samples)} train / {len(eval_samples)} eval samples")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=str(out_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()
    print(f"[TRAIN] Done. Checkpoints under: {trainer.output_path}")

    # ----------------------------------------------------------------------- #
    # Auto-export: convert the best trainer checkpoint into an inference model.
    # The GPTTrainer saves weights prefixed `xtts.` plus optimizer/training-only
    # state. An Xtts inference model needs the bare gpt.*/hifigan_decoder.* keys
    # under a top-level "model" wrapper. Without this the model loads but the GPT
    # weights silently don't match -> instant stop-token -> near-silent output.
    # ----------------------------------------------------------------------- #
    best = Path(trainer.output_path) / "best_model.pth"
    if not best.exists():
        cands = sorted(Path(trainer.output_path).glob("best_model*.pth"))
        best = cands[-1] if cands else None
    if best and best.exists():
        ck = torch.load(best, map_location="cpu", weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        new_sd = {k[len("xtts."):]: v for k, v in sd.items() if k.startswith("xtts.")}
        export_path = Path(args.output) / "model.pth"
        torch.save({"model": new_sd}, export_path)
        import shutil
        shutil.copy2(base_dir / "vocab.json", Path(args.output) / "vocab.json")
        cfg_src = Path(trainer.output_path) / "config.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, Path(args.output) / "config.json")
        print(f"[EXPORT] Inference model ready: {export_path} ({len(new_sd)} weights)")
        print(f"[EXPORT] Use model.pth + config.json + vocab.json in {args.output}")
    else:
        print("[EXPORT] WARNING: no best_model*.pth found to export.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(THIS_DIR / "alltalk" / "finetune" / "input" / "nicholas_ofczarek"))
    ap.add_argument("--base-dir", default=str(THIS_DIR / "xtts_base"))
    ap.add_argument("--output", default=str(THIS_DIR / "finetuned_models" / "nicholas_ofczarek_de"))
    ap.add_argument("--speaker", default="nicholas")
    ap.add_argument("--language", default="de")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-pct", type=float, default=0.15)
    ap.add_argument("--max-wav-length", type=int, default=255995)  # ~11.6s @22050 (XTTS default)
    ap.add_argument("--workers", type=int, default=0)  # 0 = safest on Windows
    ap.add_argument("--save-step", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prepare-only", action="store_true", help="only build CSVs + base model, no training")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    train_csv, eval_csv = prepare_csvs(dataset_dir, args.speaker, args.eval_pct, args.seed)
    base_dir = ensure_base_model(Path(args.base_dir))

    if args.prepare_only:
        print("[DONE] prepare-only: CSVs and base model ready.")
        return
    train(args, train_csv, eval_csv, base_dir)


if __name__ == "__main__":
    main()
