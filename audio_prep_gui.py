#!/usr/bin/env python3
"""
XTTS Studio — single-window GUI for the whole local voice-clone pipeline.

Tabs:
  1. Prep   — add cleaned (single-speaker) audio -> trim silence -> segment to
              <=11s -> optional faster-whisper transcription -> metadata.txt
              (no diarisation; assumes input is already one speaker).
  2. Train  — fine-tune XTTS v2 on a prepared dataset (runs train_xtts_local.py
              as a subprocess, streams its log; auto-exports an inference model).
  3. Test   — load a fine-tuned model, type German text, generate + play audio.

Heavy ML imports (TTS) are lazy-loaded only when the Test tab is used, so the
window opens instantly. Training runs in a subprocess for isolation.
"""

import os
import sys
import queue
import threading
import traceback
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import librosa
import soundfile as sf

THIS_DIR = Path(__file__).resolve().parent
PY = sys.executable  # the venv python this GUI was launched with

# Optional transcription
try:
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False


# --------------------------------------------------------------------------- #
# PREP: audio processing (worker thread)
# --------------------------------------------------------------------------- #
def process_file(src_path, out_dir, params, log, whisper_model, start_index):
    """Process one source file -> list of (out_filename, text) tuples."""
    sr = params["sample_rate"]
    top_db = params["top_db"]
    max_dur = params["max_dur"]
    min_dur = params["min_dur"]

    log(f"  loading {Path(src_path).name} ...")
    y, _ = librosa.load(src_path, sr=sr, mono=True)

    y, _ = librosa.effects.trim(y, top_db=top_db)
    if y.size == 0:
        log("  (empty after trim, skipped)")
        return []

    intervals = librosa.effects.split(y, top_db=top_db)
    max_len = int(max_dur * sr)
    min_len = int(min_dur * sr)
    pad = int(0.05 * sr)

    results = []
    idx = start_index
    for iv_start, iv_end in intervals:
        seg = y[iv_start:iv_end]
        for chunk_start in range(0, len(seg), max_len):
            chunk = seg[chunk_start:chunk_start + max_len]
            if len(chunk) < min_len:
                continue
            chunk, _ = librosa.effects.trim(chunk, top_db=top_db)
            if len(chunk) < min_len:
                continue
            chunk = np.concatenate([
                np.zeros(pad, dtype=chunk.dtype),
                chunk,
                np.zeros(pad, dtype=chunk.dtype),
            ])
            out_name = f"clip_{idx:04d}.wav"
            sf.write(os.path.join(out_dir, out_name), chunk, sr)

            text = ""
            if whisper_model is not None:
                wav_path = os.path.join(out_dir, out_name)
                log(f"    transcribing {out_name}...")
                segs, _ = whisper_model.transcribe(wav_path, language=params["lang"])
                text = " ".join(s.text.strip() for s in segs).strip()
                if text:
                    log(f"    ✓ {out_name}: {text[:60]}..." if len(text) > 60 else f"    ✓ {out_name}: {text}")
                else:
                    log(f"    {out_name}: (no transcript, kept anyway)")
            results.append((out_name, text))
            idx += 1

    log(f"  -> {len(results)} clips")
    return results


def prep_worker(files, out_dir, params, log, done, progress):
    try:
        os.makedirs(out_dir, exist_ok=True)
        whisper_model = None
        if params["transcribe"] and HAVE_WHISPER:
            log("Loading faster-whisper (medium)...")
            device = "cuda" if params["use_cuda"] else "cpu"
            compute = "float16" if params["use_cuda"] else "int8"
            try:
                whisper_model = WhisperModel("medium", device=device, compute_type=compute)
            except Exception as e:
                log(f"  whisper load failed ({e}); continuing without transcripts")
                whisper_model = None

        all_rows, index = [], 0
        for i, f in enumerate(files, 1):
            log(f"[{i}/{len(files)}] {Path(f).name}")
            try:
                rows = process_file(f, out_dir, params, log, whisper_model, index)
                all_rows.extend(rows)
                index += len(rows)
            except Exception as e:
                log(f"  ERROR: {e}")
                log(traceback.format_exc())
            progress(i / len(files) * 100)

        meta_path = os.path.join(out_dir, "metadata.txt")
        if params["transcribe"] and whisper_model is not None:
            with open(meta_path, "w", encoding="utf-8") as fh:
                for name, text in all_rows:
                    fh.write(f"{name}|{text}|{params['lang']}\n")
            log(f"\nWrote {meta_path} ({len(all_rows)} rows)")
        else:
            log("\nNo transcription performed — metadata.txt not written.")
        log(f"\nDONE. {len(all_rows)} clips in {out_dir}")
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
    finally:
        done()


# --------------------------------------------------------------------------- #
# Small helper: a log pane backed by a queue, drained on the Tk main loop
# --------------------------------------------------------------------------- #
class LogPane:
    def __init__(self, parent, height=14):
        self.q = queue.Queue()
        self.txt = tk.Text(parent, height=height, wrap="word", state="disabled",
                           bg="#111", fg="#0f0", font=("Consolas", 9))
        self.txt.pack(fill="both", expand=True, padx=8, pady=4)

    def log(self, msg):
        self.q.put(str(msg))

    def drain(self):
        while not self.q.empty():
            msg = self.q.get()
            self.txt.config(state="normal")
            self.txt.insert("end", msg + "\n")
            self.txt.see("end")
            self.txt.config(state="disabled")


# --------------------------------------------------------------------------- #
# MAIN APP
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XTTS Studio — prep · train · test")
        self.geometry("840x720")
        self._test_model = None
        self._test_model_key = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.prep_tab = ttk.Frame(nb)
        self.train_tab = ttk.Frame(nb)
        self.test_tab = ttk.Frame(nb)
        nb.add(self.prep_tab, text="1 · Prep")
        nb.add(self.train_tab, text="2 · Train")
        nb.add(self.test_tab, text="3 · Test")

        self._build_prep()
        self._build_train()
        self._build_test()
        self.after(100, self._tick)

    # ===================================================================== #
    # PREP TAB
    # ===================================================================== #
    def _build_prep(self):
        t = self.prep_tab
        pad = {"padx": 8, "pady": 4}
        self.files = []

        ttk.Label(t, text="Input audio (already cleaned / single speaker, low noise):"
                  ).pack(anchor="w", **pad)
        lf = ttk.Frame(t); lf.pack(fill="x", **pad)
        self.listbox = tk.Listbox(lf, height=6, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.listbox.yview); sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        b = ttk.Frame(t); b.pack(fill="x", **pad)
        ttk.Button(b, text="Add files…", command=self.add_files).pack(side="left")
        ttk.Button(b, text="Remove selected", command=self.remove_sel).pack(side="left", padx=4)
        ttk.Button(b, text="Clear", command=self.clear_files).pack(side="left")

        of = ttk.Frame(t); of.pack(fill="x", **pad)
        ttk.Label(of, text="Output folder:").pack(side="left")
        self.out_var = tk.StringVar(value=str(
            THIS_DIR / "alltalk" / "finetune" / "input" / "prepared_clips"))
        ttk.Entry(of, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(of, text="Browse…", command=self.pick_out).pack(side="left")

        p = ttk.LabelFrame(t, text="Settings"); p.pack(fill="x", **pad)
        self.sr_var = tk.IntVar(value=22050)
        self.topdb_var = tk.IntVar(value=30)
        self.maxdur_var = tk.DoubleVar(value=11.0)
        self.mindur_var = tk.DoubleVar(value=2.0)
        self.lang_var = tk.StringVar(value="de")

        def row(parent, label, widget):
            r = ttk.Frame(parent); r.pack(side="left", padx=10, pady=6)
            ttk.Label(r, text=label).pack(anchor="w"); widget(r)

        row(p, "Sample rate", lambda x: ttk.Combobox(
            x, width=8, textvariable=self.sr_var,
            values=[22050, 24000, 44100, 48000]).pack())
        row(p, "Silence thresh (top_db)", lambda x: ttk.Spinbox(
            x, width=6, from_=10, to=60, textvariable=self.topdb_var).pack())
        row(p, "Max clip (s)", lambda x: ttk.Spinbox(
            x, width=6, from_=3, to=15, increment=0.5, textvariable=self.maxdur_var).pack())
        row(p, "Min clip (s)", lambda x: ttk.Spinbox(
            x, width=6, from_=0.5, to=5, increment=0.5, textvariable=self.mindur_var).pack())
        row(p, "Lang", lambda x: ttk.Entry(x, width=5, textvariable=self.lang_var).pack())

        tf = ttk.Frame(t); tf.pack(fill="x", **pad)
        self.transcribe_var = tk.BooleanVar(value=HAVE_WHISPER)
        cb = ttk.Checkbutton(tf, text="Transcribe with faster-whisper -> metadata.txt",
                             variable=self.transcribe_var)
        cb.pack(side="left")
        self.prep_cuda_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tf, text="Use CUDA for whisper",
                        variable=self.prep_cuda_var).pack(side="left", padx=10)
        if not HAVE_WHISPER:
            cb.config(state="disabled")
            ttk.Label(tf, text="(faster-whisper not installed)", foreground="#a00").pack(side="left")

        rf = ttk.Frame(t); rf.pack(fill="x", **pad)
        self.prep_btn = ttk.Button(rf, text="Process", command=self.run_prep)
        self.prep_btn.pack(side="left")
        self.prep_prog = ttk.Progressbar(rf, mode="determinate", maximum=100)
        self.prep_prog.pack(side="left", fill="x", expand=True, padx=8)

        self.prep_log = LogPane(t, height=12)

    def add_files(self):
        fs = filedialog.askopenfilenames(
            title="Select cleaned audio files",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All", "*.*")])
        for f in fs:
            if f not in self.files:
                self.files.append(f); self.listbox.insert("end", f)

    def remove_sel(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i); del self.files[i]

    def clear_files(self):
        self.listbox.delete(0, "end"); self.files.clear()

    def pick_out(self):
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.out_var.set(d)

    def run_prep(self):
        if not self.files:
            messagebox.showwarning("No files", "Add some audio files first."); return
        params = {
            "sample_rate": int(self.sr_var.get()),
            "top_db": int(self.topdb_var.get()),
            "max_dur": float(self.maxdur_var.get()),
            "min_dur": float(self.mindur_var.get()),
            "lang": self.lang_var.get().strip() or "de",
            "transcribe": bool(self.transcribe_var.get()),
            "use_cuda": bool(self.prep_cuda_var.get()),
        }
        out_dir = self.out_var.get().strip()
        self.prep_btn.config(state="disabled"); self.prep_prog["value"] = 0
        # Pre-fill the Train tab's dataset with what we just produced.
        self.train_dataset_var.set(out_dir)
        threading.Thread(target=prep_worker, args=(
            list(self.files), out_dir, params, self.prep_log.log,
            lambda: self.after(0, lambda: self.prep_btn.config(state="normal")),
            lambda v: self.after(0, lambda: self.prep_prog.config(value=v))),
            daemon=True).start()

    # ===================================================================== #
    # TRAIN TAB
    # ===================================================================== #
    def _build_train(self):
        t = self.train_tab
        pad = {"padx": 8, "pady": 4}

        df = ttk.Frame(t); df.pack(fill="x", **pad)
        ttk.Label(df, text="Dataset folder (must contain metadata.txt + clips):"
                  ).pack(anchor="w")
        r = ttk.Frame(df); r.pack(fill="x")
        self.train_dataset_var = tk.StringVar(value=str(
            THIS_DIR / "alltalk" / "finetune" / "input" / "prepared_clips"))
        ttk.Entry(r, textvariable=self.train_dataset_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(r, text="Browse…", command=lambda: self._pick_into(
            self.train_dataset_var)).pack(side="left", padx=4)

        of = ttk.Frame(t); of.pack(fill="x", **pad)
        ttk.Label(of, text="Output model folder:").pack(anchor="w")
        r2 = ttk.Frame(of); r2.pack(fill="x")
        self.train_output_var = tk.StringVar(value=str(
            THIS_DIR / "finetuned_models" / "nicholas_ofczarek_de"))
        ttk.Entry(r2, textvariable=self.train_output_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(r2, text="Browse…", command=lambda: self._pick_into(
            self.train_output_var)).pack(side="left", padx=4)

        p = ttk.LabelFrame(t, text="Hyperparameters"); p.pack(fill="x", **pad)
        self.epochs_var = tk.IntVar(value=12)
        self.lr_var = tk.StringVar(value="1e-5")
        self.batch_var = tk.IntVar(value=2)
        self.grad_var = tk.IntVar(value=4)
        self.speaker_var = tk.StringVar(value="nicholas")
        self.train_lang_var = tk.StringVar(value="de")

        def row(parent, label, widget):
            r = ttk.Frame(parent); r.pack(side="left", padx=10, pady=6)
            ttk.Label(r, text=label).pack(anchor="w"); widget(r)

        row(p, "Epochs", lambda x: ttk.Spinbox(
            x, width=6, from_=1, to=100, textvariable=self.epochs_var).pack())
        row(p, "Learning rate", lambda x: ttk.Entry(
            x, width=8, textvariable=self.lr_var).pack())
        row(p, "Batch size", lambda x: ttk.Spinbox(
            x, width=5, from_=1, to=16, textvariable=self.batch_var).pack())
        row(p, "Grad accum", lambda x: ttk.Spinbox(
            x, width=5, from_=1, to=32, textvariable=self.grad_var).pack())
        row(p, "Speaker", lambda x: ttk.Entry(
            x, width=12, textvariable=self.speaker_var).pack())
        row(p, "Lang", lambda x: ttk.Entry(
            x, width=5, textvariable=self.train_lang_var).pack())

        warn = ("Note: training pins the GPU for hours. Don't run other GPU tasks "
                "(e.g. local LLMs) meanwhile. A ready-to-use model.pth is "
                "auto-exported to the output folder when done.")
        ttk.Label(t, text=warn, wraplength=800, foreground="#a60").pack(anchor="w", **pad)

        rf = ttk.Frame(t); rf.pack(fill="x", **pad)
        self.train_btn = ttk.Button(rf, text="Start training", command=self.run_train)
        self.train_btn.pack(side="left")
        self.train_stop_btn = ttk.Button(rf, text="Stop", command=self.stop_train,
                                         state="disabled")
        self.train_stop_btn.pack(side="left", padx=4)
        self._train_proc = None

        self.train_log = LogPane(t, height=16)

    def run_train(self):
        ds = self.train_dataset_var.get().strip()
        if not (Path(ds) / "metadata.txt").exists():
            messagebox.showwarning(
                "No metadata.txt",
                f"{ds}\n\nNo metadata.txt found. Run the Prep tab first "
                "(with transcription enabled).")
            return
        cmd = [PY, str(THIS_DIR / "train_xtts_local.py"),
               "--dataset", ds,
               "--output", self.train_output_var.get().strip(),
               "--epochs", str(int(self.epochs_var.get())),
               "--lr", self.lr_var.get().strip(),
               "--batch-size", str(int(self.batch_var.get())),
               "--grad-accum", str(int(self.grad_var.get())),
               "--speaker", self.speaker_var.get().strip() or "speaker",
               "--language", self.train_lang_var.get().strip() or "de"]
        self.train_log.log("LAUNCH: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        self.train_btn.config(state="disabled")
        self.train_stop_btn.config(state="normal")
        threading.Thread(target=self._train_thread, args=(cmd,), daemon=True).start()

    def _train_thread(self, cmd):
        try:
            self._train_proc = subprocess.Popen(
                cmd, cwd=str(THIS_DIR), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True,
                encoding="utf-8", errors="replace")
            for line in self._train_proc.stdout:
                self.train_log.log(line.rstrip())
            self._train_proc.wait()
            code = self._train_proc.returncode
            self.train_log.log(f"\n[TRAIN PROCESS EXITED] code={code}")
            if code == 0:
                # point the Test tab at the freshly exported model
                self.after(0, lambda: self.test_ckpt_var.set(
                    self.train_output_var.get().strip()))
        except Exception as e:
            self.train_log.log(f"ERROR launching training: {e}")
            self.train_log.log(traceback.format_exc())
        finally:
            self._train_proc = None
            self.after(0, lambda: self.train_btn.config(state="normal"))
            self.after(0, lambda: self.train_stop_btn.config(state="disabled"))

    def stop_train(self):
        if self._train_proc and self._train_proc.poll() is None:
            self._train_proc.terminate()
            self.train_log.log("[STOP] terminate signal sent.")

    # ===================================================================== #
    # TEST TAB
    # ===================================================================== #
    def _build_test(self):
        t = self.test_tab
        pad = {"padx": 8, "pady": 4}

        cf = ttk.Frame(t); cf.pack(fill="x", **pad)
        ttk.Label(cf, text="Model folder (contains model.pth + config.json + vocab.json):"
                  ).pack(anchor="w")
        r = ttk.Frame(cf); r.pack(fill="x")
        self.test_ckpt_var = tk.StringVar(value=str(
            THIS_DIR / "finetuned_models" / "nicholas_ofczarek_de"))
        ttk.Entry(r, textvariable=self.test_ckpt_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(r, text="Browse…", command=lambda: self._pick_into(
            self.test_ckpt_var)).pack(side="left", padx=4)

        sf_ = ttk.Frame(t); sf_.pack(fill="x", **pad)
        ttk.Label(sf_, text="Reference voice clip (wav, for speaker conditioning):"
                  ).pack(anchor="w")
        r2 = ttk.Frame(sf_); r2.pack(fill="x")
        self.ref_var = tk.StringVar(value=str(
            THIS_DIR / "alltalk" / "finetune" / "input" / "nicholas_clips" / "v1_utt_0000.wav"))
        ttk.Entry(r2, textvariable=self.ref_var).pack(side="left", fill="x", expand=True)
        ttk.Button(r2, text="Browse…", command=self._pick_ref).pack(side="left", padx=4)

        ttk.Label(t, text="Text to speak:").pack(anchor="w", **pad)
        self.test_text = tk.Text(t, height=4, wrap="word")
        self.test_text.pack(fill="x", **pad)
        self.test_text.insert("1.0",
            "Guten Tag, meine Damen und Herren. Ich freue mich sehr, "
            "heute hier bei Ihnen zu sein.")

        of = ttk.Frame(t); of.pack(fill="x", **pad)
        ttk.Label(of, text="Lang:").pack(side="left")
        self.test_lang_var = tk.StringVar(value="de")
        ttk.Entry(of, width=5, textvariable=self.test_lang_var).pack(side="left", padx=4)
        ttk.Label(of, text="Temp:").pack(side="left", padx=(12, 0))
        self.temp_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(of, width=6, from_=0.1, to=1.5, increment=0.05,
                    textvariable=self.temp_var).pack(side="left", padx=4)
        self.test_cuda_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(of, text="Use CUDA", variable=self.test_cuda_var).pack(
            side="left", padx=12)

        rf = ttk.Frame(t); rf.pack(fill="x", **pad)
        self.gen_btn = ttk.Button(rf, text="Generate", command=self.run_test)
        self.gen_btn.pack(side="left")
        self.play_btn = ttk.Button(rf, text="Play", command=self.play_last, state="disabled")
        self.play_btn.pack(side="left", padx=4)
        self.last_wav = None
        ttk.Label(rf, text="(output saved as test_output.wav)").pack(side="left", padx=8)

        self.test_log = LogPane(t, height=10)

    def _pick_ref(self):
        f = filedialog.askopenfilename(
            title="Reference clip", filetypes=[("Audio", "*.wav *.flac *.mp3"), ("All", "*.*")])
        if f:
            self.ref_var.set(f)

    def run_test(self):
        ckpt_dir = Path(self.test_ckpt_var.get().strip())
        ref = self.ref_var.get().strip()
        text = self.test_text.get("1.0", "end").strip()
        if not (ckpt_dir / "model.pth").exists():
            messagebox.showwarning("No model", f"model.pth not found in {ckpt_dir}")
            return
        if not Path(ref).exists():
            messagebox.showwarning("No reference", f"Reference clip not found:\n{ref}")
            return
        if not text:
            messagebox.showwarning("No text", "Type some text to speak."); return
        self.gen_btn.config(state="disabled")
        params = dict(
            ckpt_dir=str(ckpt_dir), ref=ref, text=text,
            lang=self.test_lang_var.get().strip() or "de",
            temp=float(self.temp_var.get()),
            use_cuda=bool(self.test_cuda_var.get()))
        threading.Thread(target=self._test_thread, args=(params,), daemon=True).start()

    def _test_thread(self, p):
        log = self.test_log.log
        try:
            log("Loading model (first time per session is slow)...")
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            key = (p["ckpt_dir"], p["use_cuda"])
            if self._test_model_key != key:
                ckpt = Path(p["ckpt_dir"])
                cfg_path = ckpt / "config.json"
                if not cfg_path.exists():
                    cfg_path = THIS_DIR / "xtts_base" / "config.json"
                vocab = ckpt / "vocab.json"
                if not vocab.exists():
                    vocab = THIS_DIR / "xtts_base" / "vocab.json"
                config = XttsConfig(); config.load_json(str(cfg_path))
                model = Xtts.init_from_config(config)
                model.load_checkpoint(
                    config, checkpoint_path=str(ckpt / "model.pth"),
                    vocab_path=str(vocab), use_deepspeed=False)
                if p["use_cuda"]:
                    import torch
                    if torch.cuda.is_available():
                        model.cuda()
                    else:
                        log("  CUDA not available — using CPU")
                self._test_model = model
                self._test_model_key = key
                log("  model loaded.")
            else:
                model = self._test_model

            log("Computing speaker latents...")
            g, s = model.get_conditioning_latents(audio_path=[p["ref"]])
            log("Generating...")
            out = model.inference(
                text=p["text"], language=p["lang"],
                gpt_cond_latent=g, speaker_embedding=s,
                temperature=p["temp"], enable_text_splitting=True)
            wav = np.asarray(out["wav"])
            out_path = THIS_DIR / "test_output.wav"
            sf.write(str(out_path), wav, 24000)
            self.last_wav = str(out_path)
            log(f"Done. {len(wav)/24000:.2f}s -> {out_path.name}")
            self.after(0, lambda: self.play_btn.config(state="normal"))
            self.after(0, self.play_last)
        except Exception as e:
            log(f"ERROR: {e}")
            log(traceback.format_exc())
        finally:
            self.after(0, lambda: self.gen_btn.config(state="normal"))

    def play_last(self):
        if not self.last_wav:
            return
        try:
            import winsound
            winsound.PlaySound(self.last_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            try:
                os.startfile(self.last_wav)  # fallback: default player
            except Exception as e:
                self.test_log.log(f"Could not play: {e}")

    # ===================================================================== #
    # shared
    # ===================================================================== #
    def _pick_into(self, var):
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _tick(self):
        self.prep_log.drain()
        self.train_log.drain()
        self.test_log.drain()
        self.after(100, self._tick)


if __name__ == "__main__":
    App().mainloop()
