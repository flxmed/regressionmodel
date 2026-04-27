# file: main.py (COMPLETELY REDESIGNED)
# Usage examples (PowerShell):
#   cd C:\Users\itsmi\.vscode\python
#   .\.venv\Scripts\python.exe main.py init
#   .\.venv\Scripts\python.exe main.py render scene_01
#   .\.venv\Scripts\python.exe main.py render_all
#   .\.venv\Scripts\python.exe main.py assemble narration.wav

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

# --- Voice preset ---
VOICE_SPEAKER = 5
LENGTH_SCALE = 0.95  # Reduced from 1.17 to speed up and reduce pauses
SENTENCE_SILENCE = 0.12  # Reduced from 0.16 for shorter pauses
NOISE_SCALE = 0.20
NOISE_W_SCALE = 0.20
VOLUME = 0.90
USE_CUDA = True

FFMPEG_MASTER_AF = "highpass=f=70,alimiter=limit=0.97"


# ================================================================
# PRONUNCIATION DICTIONARY - The ONLY source of truth
# ================================================================

# This dictionary maps technical terms to their phonetic spelling.
# Piper TTS works best with simple phonetic respelling.
# 
# Rules:
# 1. Use hyphens to connect syllables that should flow together
# 2. Write phonetically as you would SAY it, not as it's spelled
# 3. For acronyms: spell each letter as a short word (ay, bee, see, etc.)
# 4. Add slight pauses with commas for breath control
# 5. PLURALS FIRST - always put plural forms before singular

PRONUNCIATION_DICT: List[Tuple[str, str]] = [
    # ============ ACRONYMS (PLURAL FIRST!) ============
    # Strategy: Spell each letter phonetically as a word
    
    # AI (most common)
    (r"\bAIs\b", "AY EYE ESS"),
    (r"\bAI\b", "AY EYE"),
    
    # API
    (r"\bAPIs\b", "AY PEE EYE ESS"),
    (r"\bAPI\b", "AY PEE EYE"),
    
    # LLM
    (r"\bLLMs\b", "ELL ELL EMS"),
    (r"\bLLM\b", "ELL ELL EM"),
    
    # ML
    (r"\bML\b", "EM ELL"),
    
    # GPU
    (r"\bGPUs\b", "JEE PEE YOOS"),
    (r"\bGPU\b", "JEE PEE YOO"),
    
    # CPU
    (r"\bCPUs\b", "SEE PEE YOOS"),
    (r"\bCPU\b", "SEE PEE YOO"),
    
    # TPU
    (r"\bTPUs\b", "TEE PEE YOOS"),
    (r"\bTPU\b", "TEE PEE YOO"),
    
    # NPU
    (r"\bNPUs\b", "EN PEE YOOS"),
    (r"\bNPU\b", "EN PEE YOO"),
    
    # RAM
    (r"\bRAM\b", "AR AY EM"),
    
    # SQL
    (r"\bSQL\b", "ESS KYOO ELL"),
    
    # JSON
    (r"\bJSON\b", "JAY SON"),
    
    # YAML
    (r"\bYAML\b", "YAM ELL"),
    
    # XML
    (r"\bXML\b", "EX EM ELL"),
    
    # HTML
    (r"\bHTML\b", "AYCH TEE EM ELL"),
    
    # CSS
    (r"\bCSS\b", "SEE ESS ESS"),
    
    # HTTP/HTTPS
    (r"\bHTTPS\b", "AYCH TEE TEE PEE ESS"),
    (r"\bHTTP\b", "AYCH TEE TEE PEE"),
    
    # AWS
    (r"\bAWS\b", "AY DOUBLE-YOO ESS"),
    
    # GCP
    (r"\bGCP\b", "JEE SEE PEE"),
    
    # CI/CD
    (r"\bCI/CD\b", "SEE EYE SEE DEE"),
    (r"\bCI\b", "SEE EYE"),
    (r"\bCD\b", "SEE DEE"),
    
    # RAG
    (r"\bRAG\b", "AR AY JEE"),
    
    # SRE
    (r"\bSRE\b", "ESS AR EE"),
    
    # REST
    (r"\bREST\b", "AR EE ESS TEE"),
    
    # SaaS, PaaS, IaaS
    (r"\bSaaS\b", "SASS"),
    (r"\bPaaS\b", "PASS"),
    (r"\bIaaS\b", "EYE ASS"),
    
    # SDK
    (r"\bSDKs\b", "ESS DEE KAYS"),
    (r"\bSDK\b", "ESS DEE KAY"),
    
    # ============ TECH COMPANY/PRODUCT NAMES ============
    
    # OpenAI
    (r"\bOpenAI\b", "OPEN AY EYE"),
    
    # ChatGPT
    (r"\bChatGPT\b", "CHAT JEE PEE TEE"),
    
    # GitHub
    (r"\bGitHub\b", "GIT HUB"),
    
    # Kubernetes
    (r"\bKubernetes\b", "KOO-BER-NET-EEZ"),
    (r"\bK8s\b", "KOO-BER-NET-EEZ"),
    
    # Docker
    (r"\bDocker\b", "DOCKER"),
    
    # Redis
    (r"\bRedis\b", "RED-ISS"),
    
    # PostgreSQL / Postgres
    (r"\bPostgreSQL\b", "POST-GRES"),
    (r"\bPostgres\b", "POST-GRES"),
    
    # MongoDB
    (r"\bMongoDB\b", "MON-GO-DEE-BEE"),
    
    # MySQL
    (r"\bMySQL\b", "MY-SEE-KWEL"),
    
    # NGINX
    (r"\bNGINX\b", "ENGINE-EX"),
    (r"\bNginx\b", "ENGINE-EX"),
    
    # Apache
    (r"\bApache\b", "UH-PATCH-EE"),
    
    # ============ TECHNICAL TERMS ============
    
    # Nvidia
    (r"\bNvidia\b", "EN-VID-EE-UH"),
    (r"\bNVIDIA\b", "EN-VID-EE-UH"),
    
    # CUDA
    (r"\bCUDA\b", "KOO-DUH"),
    
    # PyTorch
    (r"\bPyTorch\b", "PIE-TORCH"),
    
    # TensorFlow
    (r"\bTensorFlow\b", "TEN-SIR-FLOW"),
    
    # Async/await
    (r"\basync\b", "AY-SINK"),
    (r"\bawait\b", "UH-WAIT"),
    
    # Cache
    (r"\bcache\b", "CASH"),
    (r"\bcaches\b", "CASHES"),
    (r"\bcached\b", "CASHED"),
    (r"\bcaching\b", "CASHING"),
    
    # Latency
    (r"\blatency\b", "LAY-TEN-SEE"),
    (r"\blatencies\b", "LAY-TEN-SEEZ"),
    
    # Throughput
    (r"\bthroughput\b", "THROO-PUT"),
    
    # Inference
    (r"\binference\b", "IN-FER-ENCE"),
    
    # Queue/queues
    (r"\bqueues\b", "KYOOZ"),
    (r"\bqueue\b", "KYOO"),
    (r"\bqueued\b", "KYOOD"),
    (r"\bqueuing\b", "KYOO-ING"),
    
    # Schema
    (r"\bschemas\b", "SKEE-MUHS"),
    (r"\bschema\b", "SKEE-MUH"),
    
    # Epoch
    (r"\bepochs\b", "EP-OCKS"),
    (r"\bepoch\b", "EP-OCK"),
    
    # Albeit
    (r"\balbeit\b", "ALL-BEE-IT"),
    
    # Via
    (r"\bvia\b", "VIE-UH"),
    
    # GUI
    (r"\bGUI\b", "GOO-EE"),
    
    # CLI
    (r"\bCLI\b", "SEE ELL EYE"),
    
    # EOF
    (r"\bEOF\b", "EE OH EFF"),
]


# ================================================================
# TEXT PROCESSING PIPELINE
# ================================================================

def fix_encoding_issues(text: str) -> str:
    """Fix common UTF-8 encoding problems."""
    replacements = {
        # Broken quotes and apostrophes
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "'": "'",
        "'": "'",
        """: '"',
        """: '"',
        "–": "-",  # en-dash
        "—": " - ",  # em-dash
        "…": "...",
        "×": " times ",
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    return text


def normalize_symbols(text: str) -> str:
    """Convert symbols to words carefully."""
    
    # Currency (only when next to numbers)
    text = re.sub(r'\$(\d)', r'dollars \1', text)
    text = re.sub(r'(\d)\$', r'\1 dollars', text)
    
    # Percentages
    text = re.sub(r'(\d)%', r'\1 percent', text)
    
    # Math operators (only when clearly used as operators)
    text = re.sub(r'\s*~\s*', ' approximately ', text)
    text = re.sub(r'\s*≈\s*', ' approximately ', text)
    text = re.sub(r'\s*<=\s*', ' less than or equal to ', text)
    text = re.sub(r'\s*>=\s*', ' greater than or equal to ', text)
    text = re.sub(r'\s*!=\s*', ' not equal to ', text)
    text = re.sub(r'\s*==\s*', ' equals ', text)
    
    # Arrows
    text = re.sub(r'\s*->\s*', ' to ', text)
    text = re.sub(r'\s*<-\s*', ' from ', text)
    text = re.sub(r'\s*=>\s*', ' leads to ', text)
    
    # Don't touch / in URLs or file paths
    # Only replace standalone slashes
    text = re.sub(r'\s+/\s+', ' or ', text)
    
    return text


def apply_pronunciation_dict(text: str) -> str:
    """Apply all pronunciation rules from dictionary."""
    for pattern, replacement in PRONUNCIATION_DICT:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    """Master text normalization function."""
    
    # Step 1: Fix encoding
    text = fix_encoding_issues(text)
    
    # Step 2: Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Step 3: Handle symbols
    text = normalize_symbols(text)
    
    # Step 4: Apply pronunciation dictionary
    text = apply_pronunciation_dict(text)
    
    # Step 5: Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def add_prosody_breaks(text: str) -> str:
    """
    Add natural pauses and breaks for better speech flow.
    Keep it minimal - Piper handles most prosody automatically.
    """
    
    # Preserve sentence boundaries
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    # Add optional breaks before discourse markers
    discourse_markers = [
        "however", "therefore", "moreover", "furthermore",
        "but", "because", "which means", "in other words",
        "for example", "for instance", "such as",
        "in fact", "indeed", "specifically"
    ]
    
    for marker in discourse_markers:
        # Add comma before marker if not already present
        pattern = rf'([^\,])\s+({re.escape(marker)})\s+'
        text = re.sub(pattern, r'\1, \2 ', text, flags=re.IGNORECASE)
    
    # Split very long lines (20+ words) for better pacing
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        words = line.split()
        if len(words) > 20:
            # Split at natural break points
            mid = len(words) // 2
            # Find comma or conjunction near middle
            for i in range(mid-5, mid+5):
                if i < len(words) and words[i].rstrip(',') in ['and', 'but', 'or', 'so']:
                    lines.append(' '.join(words[:i+1]))
                    lines.append(' '.join(words[i+1:]))
                    break
            else:
                # No natural break, just split at middle
                lines.append(' '.join(words[:mid]))
                lines.append(' '.join(words[mid:]))
        else:
            lines.append(line)
    
    return '\n'.join(lines).strip() + '\n'


# ================================================================
# CACHE & CONFIG
# ================================================================

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


# ================================================================
# TTS ENGINE
# ================================================================

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

    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def run_ffmpeg_master(raw_wav: Path, out_wav: Path, mst: MasterConfig) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(raw_wav), "-af", mst.ffmpeg_af, str(out_wav)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ================================================================
# SCENE RENDERING
# ================================================================

def render_scene(scene_name: str, eng: EngineConfig, mst: MasterConfig) -> Path:
    ensure_dirs()

    src = SCRIPTS_DIR / f"{scene_name}.txt"
    if not src.exists():
        raise FileNotFoundError(f"Missing script: {src}")

    # Read and process text
    raw_text = src.read_text(encoding="utf-8").strip()
    normalized = normalize_text(raw_text)
    with_prosody = add_prosody_breaks(normalized)

    # Check cache
    key = compute_hash(with_prosody, eng, mst)
    cache_meta = CACHE_DIR / f"{scene_name}.{key}.json"
    raw_wav = RAW_DIR / f"{scene_name}_raw.wav"
    final_wav = FINAL_DIR / f"{scene_name}.wav"

    if cache_meta.exists() and final_wav.exists():
        print(f"✓ [cache hit] {scene_name}")
        return final_wav

    # Save debug info
    cache_payload = {
        "scene": scene_name,
        "hash": key,
        "engine": asdict(eng),
        "master": asdict(mst),
        "01_raw_text": raw_text,
        "02_normalized": normalized,
        "03_final_with_prosody": with_prosody,
    }
    cache_meta.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Show what will be synthesized
    print(f"\n{'='*70}")
    print(f"RENDERING: {scene_name}")
    print(f"{'='*70}")
    print("ORIGINAL TEXT:")
    print(f"  {raw_text[:100]}..." if len(raw_text) > 100 else f"  {raw_text}")
    print()
    print("FINAL TEXT FOR TTS:")
    print(f"{'─'*70}")
    for i, line in enumerate(with_prosody.splitlines(), 1):
        if line.strip():
            print(f"  {i:2d}. {line}")
    print(f"{'='*70}\n")

    # Synthesize
    run_piper(with_prosody, raw_wav, eng)
    run_ffmpeg_master(raw_wav, final_wav, mst)

    print(f"✓ Rendered: {final_wav}\n")
    return final_wav


def list_scenes() -> List[str]:
    ensure_dirs()
    scenes = []
    for p in sorted(SCRIPTS_DIR.glob("scene_*.txt")):
        scenes.append(p.stem)
    return scenes


def assemble(output_name: str) -> Path:
    """Concatenate scene wavs into one narration."""
    ensure_dirs()
    scenes = sorted(FINAL_DIR.glob("scene_*.wav"))
    if not scenes:
        raise FileNotFoundError(f"No rendered scenes in {FINAL_DIR}")

    concat_list = OUT_DIR / "concat.txt"
    lines = [f"file '{s.as_posix()}'" for s in scenes]
    concat_list.write_text('\n'.join(lines) + '\n', encoding="utf-8")

    out_path = OUT_DIR / output_name
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"✓ Assembled: {out_path}")
    return out_path


# ================================================================
# CLI
# ================================================================

def cmd_init():
    ensure_dirs()
    example_text = (
        "This is ModelSide. Today we're auditing why LLM APIs "
        "become unaffordable at scale. AI infrastructure costs "
        "are crushing startups."
    )
    (SCRIPTS_DIR / "scene_01.txt").write_text(example_text, encoding="utf-8")
    
    print("✓ Initialized!")
    print(f"  Created: {SCRIPTS_DIR / 'scene_01.txt'}")
    print()
    print("Next steps:")
    print("  1. Edit scripts/scene_01.txt")
    print("  2. Run: python main.py render scene_01")


def main():
    parser = argparse.ArgumentParser(
        prog="modelside_tts",
        description="Professional TTS for technical content"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize project structure")
    
    p_render = sub.add_parser("render", help="Render a single scene")
    p_render.add_argument("scene_name", help="e.g. scene_01")

    sub.add_parser("render_all", help="Render all scenes")

    p_asm = sub.add_parser("assemble", help="Combine scenes into one file")
    p_asm.add_argument("output", help="e.g. narration.wav")

    args = parser.parse_args()

    eng = EngineConfig()
    mst = MasterConfig()

    try:
        if args.cmd == "init":
            cmd_init()
        
        elif args.cmd == "render":
            render_scene(args.scene_name, eng, mst)
        
        elif args.cmd == "render_all":
            scenes = list_scenes()
            if not scenes:
                raise SystemExit("No scripts found. Run 'init' first.")
            for s in scenes:
                render_scene(s, eng, mst)
        
        elif args.cmd == "assemble":
            assemble(args.output)
    
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()
