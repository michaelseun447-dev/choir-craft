"""
AI Choir Learning Platform - Backend API
==========================================
FastAPI application exposing endpoints to:

  POST /api/songs/upload
      Upload a song. Kicks off the full analysis pipeline:
        1. (optional) Vocal/instrumental separation (Demucs)
        2. Key, chord, melody, tempo detection (librosa)
        3. SATB harmony generation (music theory engine)
        4. Lyrics transcription + timing (Whisper)
        5. Synthesize audio reference tracks for Soprano/Alto/Tenor
      Returns a song_id and the full analysis result.

  GET /api/songs/{song_id}
      Retrieve the stored analysis result (key, chords, melody,
      harmony parts, lyrics) for a previously uploaded song.

  GET /api/songs/{song_id}/audio/{part}
      Stream the rendered audio for a given part:
        part ∈ {"original", "instrumental", "vocals",
                "soprano", "alto", "tenor"}
      Optional query param ?speed=0.5 for slowed-down practice playback.

  GET /api/songs
      List all uploaded songs (id, filename, key, duration).

DEPLOYMENT NOTES
----------------
- Heavy steps (separation, transcription) are wrapped in try/except
  so the API still returns useful results (key/chords/melody/harmony)
  even if those optional models aren't available in the runtime
  environment. In production, run those steps as background jobs and
  use websockets/polling to notify the frontend when ready.
- Uploaded files and generated audio are stored under ./storage/
  Replace with S3/cloud storage for production.
"""

import os
import uuid
import json
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from analysis import analyze_audio
from harmony import generate_harmony
from synthesis import render_voice_part, save_wav, change_playback_speed
import numpy as np
from scipy.io import wavfile

try:
    from separation import separate_vocals, is_demucs_available
except ImportError:
    is_demucs_available = lambda: False

try:
    from lyrics import transcribe_with_timing, align_lyrics_to_voice_part
    LYRICS_AVAILABLE = True
except ImportError:
    LYRICS_AVAILABLE = False


STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title="AI Choir Learning Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _song_dir(song_id: str) -> str:
    return os.path.join(STORAGE_DIR, song_id)


def _save_result(song_id: str, result: dict):
    path = os.path.join(_song_dir(song_id), "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def _load_result(song_id: str) -> dict:
    path = os.path.join(_song_dir(song_id), "result.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Song not found")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/songs/upload")
async def upload_song(
    file: UploadFile = File(...),
    separate: bool = Query(default=False, description="Run vocal/instrumental separation (slow)"),
    transcribe: bool = Query(default=False, description="Run lyrics transcription (slow)"),
):
    """
    Upload a song and run the full analysis + harmony generation pipeline.
    """
    song_id = str(uuid.uuid4())
    song_dir = _song_dir(song_id)
    os.makedirs(song_dir, exist_ok=True)

    # Save uploaded file
    original_path = os.path.join(song_dir, f"original_{file.filename}")
    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ---- Step 1: Core music analysis (key, chords, melody, tempo) ----
    try:
        analysis_result = analyze_audio(original_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {e}")

    # ---- Step 2: SATB harmony generation ----
    harmony_parts = generate_harmony(
        analysis_result["melody_notes"],
        analysis_result["chords"],
        analysis_result["key"],
    )

    # ---- Step 3 (optional): Vocal/instrumental separation ----
    separation_info = {"available": False}
    if separate and is_demucs_available():
        try:
            stems = separate_vocals(original_path, song_dir)
            separation_info = {
                "available": True,
                "vocals": stems["vocals"],
                "instrumental": stems["instrumental"],
            }
        except Exception as e:
            separation_info = {"available": False, "error": str(e)}

    # ---- Step 4 (optional): Lyrics transcription ----
    lyrics_data = {"available": False}
    vocal_track_for_transcription = separation_info.get("vocals", original_path)
    if transcribe and LYRICS_AVAILABLE:
        try:
            transcript = transcribe_with_timing(vocal_track_for_transcription)
            lyrics_data = {"available": True, **transcript}

            # attach lyric words to each harmony part's notes
            for voice in ["soprano", "alto", "tenor"]:
                harmony_parts[voice] = align_lyrics_to_voice_part(
                    transcript["words"], harmony_parts[voice]
                )
        except Exception as e:
            lyrics_data = {"available": False, "error": str(e)}

    # ---- Step 5: Synthesize reference audio for each voice part ----
    duration = analysis_result["duration_seconds"]
    audio_dir = os.path.join(song_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for voice in ["soprano", "alto", "tenor"]:
        buf = render_voice_part(harmony_parts[voice], duration, voice_name=voice)
        save_wav(buf, os.path.join(audio_dir, f"{voice}.wav"))

    # ---- Assemble result ----
    result = {
        "song_id": song_id,
        "filename": file.filename,
        "duration_seconds": duration,
        "key": analysis_result["key"],
        "tempo": analysis_result["tempo"],
        "chords": analysis_result["chords"],
        "melody_notes": analysis_result["melody_notes"],
        "harmony": harmony_parts,
        "separation": {"available": separation_info.get("available", False)},
        "lyrics": {
            "available": lyrics_data.get("available", False),
            "text": lyrics_data.get("text", ""),
            "segments": lyrics_data.get("segments", []),
        },
        "original_filename": os.path.basename(original_path),
    }

    _save_result(song_id, result)

    return JSONResponse(result)


@app.get("/api/songs")
async def list_songs():
    """List all uploaded songs with basic info."""
    songs = []
    if not os.path.exists(STORAGE_DIR):
        return {"songs": []}

    for song_id in os.listdir(STORAGE_DIR):
        result_path = os.path.join(STORAGE_DIR, song_id, "result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            songs.append({
                "song_id": data["song_id"],
                "filename": data["filename"],
                "key": data["key"]["name"],
                "duration_seconds": data["duration_seconds"],
            })

    return {"songs": songs}


@app.get("/api/songs/{song_id}")
async def get_song(song_id: str):
    """Retrieve the full analysis result for a song."""
    return _load_result(song_id)


@app.get("/api/songs/{song_id}/audio/{part}")
async def get_audio(song_id: str, part: str, speed: float = Query(default=1.0, ge=0.25, le=2.0)):
    """
    Stream audio for a given part, optionally at a slowed/sped-up rate.

    part ∈ {"original", "instrumental", "vocals", "soprano", "alto", "tenor"}
    speed: 0.25 - 2.0 (1.0 = normal speed)
    """
    song_dir = _song_dir(song_id)
    if not os.path.exists(song_dir):
        raise HTTPException(status_code=404, detail="Song not found")

    result = _load_result(song_id)

    if part in ("soprano", "alto", "tenor"):
        filepath = os.path.join(song_dir, "audio", f"{part}.wav")
    elif part == "original":
        filepath = os.path.join(song_dir, result["original_filename"])
    elif part in ("vocals", "instrumental"):
        # only available if separation was run
        model_dir = os.path.join(song_dir, "htdemucs")
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"{part} not available (separation not run)")
        # find the stem file
        for root, _, files in os.walk(model_dir):
            target = "vocals.wav" if part == "vocals" else "no_vocals.wav"
            if target in files:
                filepath = os.path.join(root, target)
                break
        else:
            raise HTTPException(status_code=404, detail=f"{part} file not found")
    else:
        raise HTTPException(status_code=400, detail="Invalid part name")

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    if speed == 1.0:
        return FileResponse(filepath, media_type="audio/wav")

    # render a speed-adjusted temporary copy
    sr, data = wavfile.read(filepath)
    float_data = data.astype(np.float32) / 32768.0
    stretched = change_playback_speed(float_data, speed)

    tmp_path = os.path.join(song_dir, "audio", f"{part}_speed_{speed}.wav")
    save_wav(stretched, tmp_path, sr=sr)

    return FileResponse(tmp_path, media_type="audio/wav")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "demucs_available": is_demucs_available(),
        "whisper_available": LYRICS_AVAILABLE,
    }
