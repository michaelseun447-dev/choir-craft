"""
Music Analysis Module
======================
Detects key, chords, melody (pitch contour), tempo and beat grid
from an audio file using librosa's signal-processing tools.

This is "real" music information retrieval (MIR) - no ML black box,
just established algorithms:
  - Chroma features -> Krumhansl-Schmuckler key estimation
  - Chroma + template matching -> chord detection
  - pYIN -> monophonic pitch tracking for the lead melody
  - Beat tracking -> tempo & beat grid for syncing lyrics
"""

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Key detection (Krumhansl-Schmuckler algorithm)
# ---------------------------------------------------------------------------

# Krumhansl-Kessler key profiles (major and minor)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(chroma_mean: np.ndarray):
    """
    Estimate the musical key from an averaged chroma vector.

    Returns: dict with 'tonic' (note name), 'mode' ('major'/'minor'),
             and 'name' (e.g. "C major")
    """
    best_corr = -2
    best_key = (0, 'major')

    for shift in range(12):
        major_template = np.roll(MAJOR_PROFILE, shift)
        minor_template = np.roll(MINOR_PROFILE, shift)

        major_corr = np.corrcoef(chroma_mean, major_template)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_template)[0, 1]

        if major_corr > best_corr:
            best_corr = major_corr
            best_key = (shift, 'major')
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = (shift, 'minor')

    tonic_idx, mode = best_key
    tonic = NOTE_NAMES[tonic_idx]
    return {
        "tonic": tonic,
        "tonic_index": tonic_idx,
        "mode": mode,
        "name": f"{tonic} {mode}",
        "confidence": float(best_corr),
    }


# ---------------------------------------------------------------------------
# Chord detection (template matching on chroma frames)
# ---------------------------------------------------------------------------

def _build_chord_templates():
    """Build binary chroma templates for major and minor triads in all 12 roots."""
    templates = {}
    for root in range(12):
        major = np.zeros(12)
        for interval in (0, 4, 7):  # root, major third, fifth
            major[(root + interval) % 12] = 1
        templates[f"{NOTE_NAMES[root]}"] = major  # major chord (no suffix)

        minor = np.zeros(12)
        for interval in (0, 3, 7):  # root, minor third, fifth
            minor[(root + interval) % 12] = 1
        templates[f"{NOTE_NAMES[root]}m"] = minor

    return templates


CHORD_TEMPLATES = _build_chord_templates()


def detect_chords(chroma: np.ndarray, sr: int, hop_length: int,
                   beat_frames: np.ndarray, segment_seconds: float = 0.5):
    """
    Detect chord per time segment using template matching.

    Parameters
    ----------
    chroma : np.ndarray, shape (12, n_frames)
    beat_frames : np.ndarray of frame indices for beats (used to align segments)

    Returns
    -------
    list of dicts: [{ "time": float, "chord": str }, ...]
    """
    n_frames = chroma.shape[1]
    frames_per_segment = max(1, int(librosa.time_to_frames(segment_seconds, sr=sr, hop_length=hop_length)))

    chords = []
    for start in range(0, n_frames, frames_per_segment):
        end = min(start + frames_per_segment, n_frames)
        segment_chroma = chroma[:, start:end].mean(axis=1)

        # normalize
        norm = np.linalg.norm(segment_chroma)
        if norm > 0:
            segment_chroma = segment_chroma / norm

        best_score = -1
        best_chord = "N"  # "no chord" fallback
        for name, template in CHORD_TEMPLATES.items():
            t_norm = template / np.linalg.norm(template)
            score = float(np.dot(segment_chroma, t_norm))
            if score > best_score:
                best_score = score
                best_chord = name

        time = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
        chords.append({"time": float(time), "chord": best_chord, "confidence": best_score})

    # Collapse consecutive duplicate chords into spans
    collapsed = []
    for c in chords:
        if collapsed and collapsed[-1]["chord"] == c["chord"]:
            continue
        collapsed.append(c)

    return collapsed


# ---------------------------------------------------------------------------
# Melody extraction (monophonic pitch tracking via pYIN)
# ---------------------------------------------------------------------------

def extract_melody(y: np.ndarray, sr: int, fmin="C2", fmax="C7", hop_length=512):
    """
    Extract the lead melody pitch contour using the pYIN algorithm.

    Returns
    -------
    dict with:
      - times: np.ndarray of frame times (seconds)
      - f0: np.ndarray of fundamental frequencies (Hz), NaN where unvoiced
      - voiced_flag: np.ndarray of booleans
      - notes: list of discrete note events [{start, end, pitch_name, midi}, ...]
    """
    fmin_hz = librosa.note_to_hz(fmin)
    fmax_hz = librosa.note_to_hz(fmax)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin_hz, fmax=fmax_hz, sr=sr, hop_length=hop_length
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Convert continuous f0 contour into discrete note events
    notes = _f0_to_notes(times, f0, voiced_flag)

    return {
        "times": times,
        "f0": f0,
        "voiced_flag": voiced_flag,
        "notes": notes,
    }


def _f0_to_notes(times, f0, voiced_flag, min_note_duration=0.08):
    """
    Convert a frame-wise f0 contour into discrete note events by
    grouping consecutive frames that map to the same MIDI pitch.
    """
    notes = []
    current_midi = None
    current_start = None

    for i, t in enumerate(times):
        if not voiced_flag[i] or np.isnan(f0[i]):
            midi = None
        else:
            midi = int(round(librosa.hz_to_midi(f0[i])))

        if midi != current_midi:
            # close previous note
            if current_midi is not None:
                duration = t - current_start
                if duration >= min_note_duration:
                    notes.append({
                        "start": float(current_start),
                        "end": float(t),
                        "midi": current_midi,
                        "pitch_name": librosa.midi_to_note(current_midi),
                    })
            current_midi = midi
            current_start = t

    # close final note
    if current_midi is not None and len(times) > 0:
        end_t = times[-1]
        duration = end_t - current_start
        if duration >= min_note_duration:
            notes.append({
                "start": float(current_start),
                "end": float(end_t),
                "midi": current_midi,
                "pitch_name": librosa.midi_to_note(current_midi),
            })

    return notes


# ---------------------------------------------------------------------------
# Tempo & beat tracking
# ---------------------------------------------------------------------------

def detect_tempo_and_beats(y: np.ndarray, sr: int):
    """Return estimated tempo (BPM) and beat times (seconds)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return {
        "tempo_bpm": float(tempo if np.ndim(tempo) == 0 else tempo[0]),
        "beat_times": beat_times.tolist(),
        "beat_frames": beat_frames,
    }


# ---------------------------------------------------------------------------
# Master analysis function
# ---------------------------------------------------------------------------

def analyze_audio(filepath: str, hop_length: int = 512):
    """
    Run full analysis pipeline on an audio file.

    Returns a dict with key, chords, melody, tempo/beats.
    """
    y, sr = librosa.load(filepath, sr=None, mono=True)

    # Chroma (for key + chords)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = chroma.mean(axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    key_info = detect_key(chroma_mean)

    tempo_info = detect_tempo_and_beats(y, sr)

    chords = detect_chords(chroma, sr, hop_length, tempo_info["beat_frames"])

    melody = extract_melody(y, sr, hop_length=hop_length)

    duration = float(len(y) / sr)

    return {
        "duration_seconds": duration,
        "sample_rate": sr,
        "key": key_info,
        "tempo": {
            "bpm": tempo_info["tempo_bpm"],
            "beat_times": tempo_info["beat_times"],
        },
        "chords": chords,
        "melody_notes": melody["notes"],
    }
