"""
Vocal Separation Module
========================
Uses Demucs (Facebook Research's source separation model) to split an
uploaded song into:
  - vocals.wav   (the original lead vocal track)
  - instrumental.wav (everything else: drums, bass, other)

NOTE ON DEPLOYMENT
------------------
Demucs models are large (~80-300MB depending on model) and separation
is CPU/GPU-intensive (can take 30s-2min per song on CPU, a few seconds
on GPU). For a production deployment this should run either:
  - on a server with a GPU (recommended), or
  - as an async background job (Celery/RQ) so the user isn't blocked

This module wraps the `demucs` CLI/python API. On first run it will
download the pretrained model weights (requires internet access).
"""

import os
import subprocess
import tempfile
import shutil


def separate_vocals(input_path: str, output_dir: str, model_name: str = "htdemucs"):
    """
    Run Demucs to separate vocals from the instrumental.

    Parameters
    ----------
    input_path : path to the uploaded song (wav/mp3/etc)
    output_dir : directory where separated stems will be placed
    model_name : demucs model variant. "htdemucs" is the default
                  high-quality hybrid transformer model.

    Returns
    -------
    dict with paths: {"vocals": <path>, "instrumental": <path>}

    Raises
    ------
    RuntimeError if demucs fails (e.g. no internet to fetch weights,
    or out of memory).
    """
    os.makedirs(output_dir, exist_ok=True)

    # demucs writes output to: <output_dir>/<model_name>/<track_name>/{vocals,drums,bass,other}.wav
    cmd = [
        "python3", "-m", "demucs.separate",
        "-n", model_name,
        "--two-stems", "vocals",  # produces vocals.wav + no_vocals.wav (instrumental)
        "-o", output_dir,
        input_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs separation failed: {e.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Demucs separation timed out (>10 minutes)")

    track_name = os.path.splitext(os.path.basename(input_path))[0]
    stem_dir = os.path.join(output_dir, model_name, track_name)

    vocals_path = os.path.join(stem_dir, "vocals.wav")
    instrumental_path = os.path.join(stem_dir, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Expected output not found at {vocals_path}")

    return {
        "vocals": vocals_path,
        "instrumental": instrumental_path,
    }


def is_demucs_available() -> bool:
    """Quick check whether demucs CLI is importable (model weights may still need downloading)."""
    try:
        import demucs.separate  # noqa
        return True
    except ImportError:
        return False
