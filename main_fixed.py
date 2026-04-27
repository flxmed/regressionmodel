# file: modelside_tts.py (FIXED VERSION)
# Usage examples (PowerShell):
#   cd C:\Users\itsmi\.vscode\python
#   .\.venv\Scripts\python.exe modelside_tts.py init
#   .\.venv\Scripts\python.exe modelside_tts.py render scene_01
#   .\.venv\Scripts\python.exe modelside_tts.py render_all
#   .\.venv\Scripts\python.exe modelside_tts.py assemble narration.wav

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent

SCRIPTS_DIR = ROOT / "scripts"
VOICES_DIR = ROOT / "voices"
OUT_DIR = ROOT / "out"
CACHE_DIR = ROOT / "work" / "cache"

RAW_DIR = OUT_DIR / "raw"
FINAL_DIR = OUT_DIR / "final"

PY_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
MODEL = VOICES_DIR / "en_US-libritts-high.onnx"
CONFIG = VOICES_DIR / "en_US-libritts-high.onnx.json"

# --- Voice preset (ModelSide baseline) ---
VOICE_SPEAKER = 5
LENGTH_SCALE = 1.17
SENTENCE_SILENCE = 0.16
NOISE_SCALE = 0.20
NOISE_W_SCALE = 0.20
VOLUME = 0.90
USE_CUDA = True

# Mastering: keep it minimal to avoid "underwater" artifacts
FFMPEG_MASTER_AF = "highpass=f=70,alimiter=limit=0.97"


# -------------------------
# Text pipeline (FIXED v2)
# -------------------------

# Custom pronunciations - INCLUDING acronyms
LEXICON_REWRITES: List[Tuple[str, str]] = [
    # Acronyms - spell them phonetically as connected words
    (r"\bLLMs\b", "el-el-emz"),
    (r"\bLLM\b", "el-el-em"),
    (r"\bAPIs\b", "ay-pee-eyez"), 
    (r"\bAPI\b", "ay-pee-eye"),
    (r"\bGPUs\b", "jee-pee-yooz"),
    (r"\bGPU\b", "jee-pee-you"),
    (r"\bCPUs\b", "see-pee-yooz"),
    (r"\bCPU\b", "see-pee-you"),
    (r"\bTPUs\b", "tee-pee-yooz"),
    (r"\bTPU\b", "tee-pee-you"),
    (r"\bSQL\b", "ess-cue-el"),
    (r"\bJSON\b", "jay-sawn"),
    (r"\bYAML\b", "yam-el"),
    (r"\bRAG\b", "ar-ay-jee"),
    (r"\bSRE\b", "ess-ar-ee"),
    
    # Technical terms
    (r"\bKubernetes\b", "koo-ber-net-eez"),
    (r"\bPostgres\b", "post-grez"),
    (r"\bPostgreSQL\b", "post-grez-cue-el"),
    (r"\bRedis\b", "red-iss"),
    (r"\bthroughput\b", "thru-put"),
    (r"\binference\b", "in-fer-ence"),
]

DISCOURSE_BREAKS = [
    "but", "because", "which means", "in other words", 
    "the catch is", "the real problem is"
]

def normalize_text(text: str) -> str:
    """FIXED v2: Simpler normalization, lexicon handles acronyms"""
    t = text.strip()

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t)
    
    # Fix common UTF-8 quote issues
    t = t.replace("'", "'")  
    t = t.replace("'", "'")  
    t = t.replace(""", '"')
    t = t.replace(""", '"')
    t = t.replace("â€™", "'")

    # Currency - only if followed by numbers
    t = re.sub(r'\$(\d)', r'dollars \1', t)
    
    # Math operators - ONLY in math context
    t = re.sub(r'\s~\s', ' approximately ', t)
    t = re.sub(r'\s<=\s', ' less than or equal to ', t)
    t = re.sub(r'\s>=\s', ' greater than or equal to ', t)

    # Lexicon rewrites (handles acronyms AND technical terms)
    # ORDER MATTERS: do plurals before singulars!
    for pat, rep in LEXICON_REWRITES:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)

    # Cleanup spacing
    t = re.sub(r"\s+", " ", t).strip()
    return t


def prosody_plan(text: str) -> str:
    """
    FIXED: Gentler prosody control
    """
    t = text.strip()

    # Only add breaks before discourse markers if they start a clause
    for marker in DISCOURSE_BREAKS:
        # Add comma before marker when preceded by space
        pattern = rf'(\w)\s+({re.escape(marker)})\s+'
        t = re.sub(pattern, r'\1, \2 ', t, flags=re.IGNORECASE)

    # Preserve existing sentence breaks
    t = re.sub(r"([.!?])\s+", r"\1\n", t)

    # Gentle line chunking (avoid very long lines)
    lines = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
            
        words = line.split()
        # Only split if line is very long (15+ words)
        if len(words) > 15:
            while len(words) > 12:
                lines.append(" ".join(words[:10]))
                words = words[10:]
        
        if words:
            lines.append(" ".join(words))
    
    return "\n".join(lines).strip() + "\n"


# -------------------------
# Cache + hashing
# -------------------------

@dataclass(frozen=True)
class EngineConfig:
    speaker: int = VOICE_SPEAKER
    length_scale: float = LENGTH_SCALE
    sentence_silence: float = SENTENCE_SILENCE
    noise_scale: float = NOISE_SCALE
    noise_w_scale: float = NOISE_W_SCALE
    volume: float = VOLUME
    use_cuda: bool = USE_CUDA
    no_normalize: bool = True


@dataclass(frozen=True)
class MasterConfig:
    ffmpeg_af: str = FFMPEG_MASTER_AF


def compute_hash(scene_text: str, eng: EngineConfig, mst: MasterConfig) -> str:
    h = hashlib.sha256()
    h.update(scene_text.encode("utf-8"))
    h.update(json.dumps(asdict(eng), sort_keys=True).encode("utf-8"))
    h.update(json.dumps(asdict(mst), sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def ensure_dirs():
    SCRIPTS_DIR.mkdir(exist_ok=True)
    VOICES_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# TTS + mastering
# -------------------------

def run_piper(text: str, out_wav: Path, eng: EngineConfig) -> None:
    if not PY_EXE.exists():
        raise FileNotFoundError(f"Python venv not found: {PY_EXE}")
    if not MODEL.exists() or not CONFIG.exists():
        raise FileNotFoundError(f"Voice model/config missing in: {VOICES_DIR}")

    cmd = [
        str(PY_EXE), "-m", "piper",
        "--model", str(MODEL),
        "--config", str(CONFIG),
        "--speaker", str(eng.speaker),
        "--length_scale", str(eng.length_scale),
        "--sentence_silence", str(eng.sentence_silence),
        "--noise_scale", str(eng.noise_scale),
        "--noise_w_scale", str(eng.noise_w_scale),
        "--volume", str(eng.volume),
        "--output_file", str(out_wav),
    ]
    if eng.use_cuda:
        cmd.append("--cuda")
    if eng.no_normalize:
        cmd.append("--no-normalize")

    # Silence ORT warnings
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def run_ffmpeg_master(raw_wav: Path, out_wav: Path, mst: MasterConfig) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(raw_wav), "-af", mst.ffmpeg_af, str(out_wav)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_scene(scene_name: str, eng: EngineConfig, mst: MasterConfig) -> Path:
    ensure_dirs()

    src = SCRIPTS_DIR / f"{scene_name}.txt"
    if not src.exists():
        raise FileNotFoundError(f"Missing script: {src}")

    raw_text = src.read_text(encoding="utf-8").strip()
    norm = normalize_text(raw_text)
    planned = prosody_plan(norm)

    key = compute_hash(planned, eng, mst)

    cache_meta = CACHE_DIR / f"{scene_name}.{key}.json"
    raw_wav = RAW_DIR / f"{scene_name}_raw.wav"
    final_wav = FINAL_DIR / f"{scene_name}.wav"

    if cache_meta.exists() and final_wav.exists():
        print(f"[cache hit] {scene_name} -> {final_wav}")
        return final_wav

    # Save cache meta for debugging
    cache_payload = {
        "scene": scene_name,
        "hash": key,
        "engine": asdict(eng),
        "master": asdict(mst),
        "raw_text": raw_text,
        "normalized_text": norm,
        "prosody_text": planned,
    }
    cache_meta.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # DEBUG: Print what will be synthesized
    print(f"\n[DEBUG] Synthesizing text for {scene_name}:")
    print("=" * 60)
    print(planned)
    print("=" * 60 + "\n")

    run_piper(planned, raw_wav, eng)
    run_ffmpeg_master(raw_wav, final_wav, mst)

    print(f"[rendered] {scene_name} -> {final_wav}")
    return final_wav


def list_scenes() -> List[str]:
    ensure_dirs()
    scenes = []
    for p in sorted(SCRIPTS_DIR.glob("scene_*.txt")):
        scenes.append(p.stem)
    return scenes


def assemble(output_name: str) -> Path:
    """Concatenate scene wavs into one narration"""
    ensure_dirs()
    scenes = sorted(FINAL_DIR.glob("scene_*.wav"))
    if not scenes:
        raise FileNotFoundError(f"No rendered scenes in {FINAL_DIR}")

    concat_list = OUT_DIR / "concat.txt"
    lines = []
    for s in scenes:
        lines.append(f"file '{s.as_posix()}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path = OUT_DIR / output_name
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[assembled] -> {out_path}")
    return out_path


def cmd_init():
    ensure_dirs()
    # FIXED: Use proper apostrophe
    (SCRIPTS_DIR / "scene_01.txt").write_text(
        "This is ModelSide. Today we're auditing why LLM APIs become unaffordable at scale.",
        encoding="utf-8",
    )
    print("Initialized:")
    print(f"  - {SCRIPTS_DIR}\\scene_01.txt (example)")
    print("Next: edit scripts/scene_01.txt, then run: render scene_01")


def main():
    parser = argparse.ArgumentParser(prog="modelside_tts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    p_render = sub.add_parser("render")
    p_render.add_argument("scene_name", help="e.g. scene_01")

    sub.add_parser("render_all")

    p_asm = sub.add_parser("assemble")
    p_asm.add_argument("output", help="e.g. narration.wav")

    args = parser.parse_args()

    eng = EngineConfig()
    mst = MasterConfig()

    if args.cmd == "init":
        cmd_init()
        return

    if args.cmd == "render":
        render_scene(args.scene_name, eng, mst)
        return

    if args.cmd == "render_all":
        scenes = list_scenes()
        if not scenes:
            raise SystemExit("No scripts found in scripts/. Create scripts/scene_01.txt etc.")
        for s in scenes:
            render_scene(s, eng, mst)
        return

    if args.cmd == "assemble":
        assemble(args.output)
        return


if __name__ == "__main__":
    main()
