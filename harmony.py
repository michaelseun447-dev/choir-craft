"""
Harmony Generation Engine
==========================
This is the "AI choir teacher" core. Given:
  - the detected musical key
  - the detected chord progression
  - the lead melody (assumed to be the Soprano line, or close to it)

...it generates SATB-style harmony parts (Soprano, Alto, Tenor, Bass-omitted
per spec but we keep the engine SATB-capable) using real choral arranging
rules:

  1. For each melody note, identify the underlying chord (from the chord
     progression at that time).
  2. Build the chord's pitch classes (root, third, fifth, [seventh]).
  3. Assign each voice (Soprano/Alto/Tenor) the closest available chord
     tone to its previous note, respecting:
       - voice ranges (Soprano, Alto, Tenor)
       - smooth voice leading (minimal movement between consecutive notes)
       - avoiding doublings of the melody when a different chord tone fits
       - keeping voices in their correct relative order (S > A > T)

This produces harmonies that are music-theoretically grounded - not
arbitrary intervals - while being deterministic and fast (no ML needed
for this stage).
"""

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Voice ranges (MIDI note numbers) - typical SATB choir ranges
# ---------------------------------------------------------------------------

VOICE_RANGES = {
    "soprano": (60, 81),  # C4 - A5
    "alto":    (53, 74),  # F3 - D5
    "tenor":   (48, 69),  # C3 - A4
}

# ---------------------------------------------------------------------------
# Chord -> pitch class construction
# ---------------------------------------------------------------------------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def chord_to_pitch_classes(chord_symbol: str):
    """
    Convert a chord symbol like "C", "Am", "G", "Dm" into a set of
    pitch classes (0-11) representing root, third, fifth.

    "N" (no chord detected) returns an empty list, in which case
    callers should fall back to the diatonic scale of the key.
    """
    if chord_symbol == "N" or not chord_symbol:
        return []

    is_minor = chord_symbol.endswith("m")
    root_name = chord_symbol[:-1] if is_minor else chord_symbol

    if root_name not in NOTE_NAMES:
        return []

    root = NOTE_NAMES.index(root_name)
    third_interval = 3 if is_minor else 4
    fifth_interval = 7

    return [root, (root + third_interval) % 12, (root + fifth_interval) % 12]


def diatonic_pitch_classes(key_info: dict):
    """Return the 7 pitch classes of the major/minor scale for the given key."""
    tonic = key_info["tonic_index"]
    if key_info["mode"] == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        # natural minor
        intervals = [0, 2, 3, 5, 7, 8, 10]
    return [(tonic + i) % 12 for i in intervals]


# ---------------------------------------------------------------------------
# Voice-leading helper
# ---------------------------------------------------------------------------

def _closest_pitch_in_range(target_pc_options, prev_midi, voice_range):
    """
    Given a set of candidate pitch classes, find the MIDI note within
    `voice_range` that:
      1. Belongs to one of the candidate pitch classes
      2. Is closest to `prev_midi` (smooth voice leading)

    If prev_midi is None (first note), prefer the middle of the range.
    """
    low, high = voice_range
    candidates = []

    for pc in target_pc_options:
        # generate every octave of this pitch class within the voice range
        for octave_base in range(0, 128, 12):
            midi = octave_base + pc
            if low <= midi <= high:
                candidates.append(midi)

    if not candidates:
        # fallback: just clip to range center
        return (low + high) // 2

    if prev_midi is None:
        # choose candidate closest to the center of the range
        center = (low + high) / 2
        return min(candidates, key=lambda m: abs(m - center))

    return min(candidates, key=lambda m: abs(m - prev_midi))


def _chord_for_time(chords, t):
    """Find the chord symbol active at time t (seconds)."""
    active = "N"
    for c in chords:
        if c["time"] <= t:
            active = c["chord"]
        else:
            break
    return active


# ---------------------------------------------------------------------------
# Main harmony generation
# ---------------------------------------------------------------------------

def generate_harmony(melody_notes, chords, key_info):
    """
    Generate Soprano, Alto, Tenor parts from the lead melody + chords.

    Parameters
    ----------
    melody_notes : list of {start, end, midi, pitch_name}
        The detected lead vocal melody (assumed to function as the
        top voice / melody line that the choir sings in unison or
        that the Soprano doubles).
    chords : list of {time, chord}
    key_info : dict from analysis.detect_key()

    Returns
    -------
    dict with keys "soprano", "alto", "tenor" - each a list of note
    events: [{start, end, midi, pitch_name}, ...]

    Approach
    --------
    - Soprano part doubles the melody (the most common choral approach:
      the choir's top line sings the tune).
    - Alto and Tenor are derived per melody note by:
        1. Determining the active chord at that note's start time.
        2. Getting that chord's pitch classes (root/3rd/5th), or
           falling back to the diatonic scale of the key if no chord
           was detected.
        3. Picking the chord tone (other than the melody's own pitch
           class, when possible) closest to the previous note in that
           voice - for smooth voice leading.
    """
    soprano = []
    alto = []
    tenor = []

    prev_alto_midi = None
    prev_tenor_midi = None

    diatonic_pcs = diatonic_pitch_classes(key_info)

    for note in melody_notes:
        start, end, melody_midi = note["start"], note["end"], note["midi"]
        melody_pc = melody_midi % 12

        chord_symbol = _chord_for_time(chords, start)
        chord_pcs = chord_to_pitch_classes(chord_symbol)

        if not chord_pcs:
            chord_pcs = diatonic_pcs

        # --- Soprano: doubles the melody exactly ---
        soprano.append({
            "start": start, "end": end,
            "midi": melody_midi,
            "pitch_name": note["pitch_name"],
        })

        # --- Alto: pick a chord tone below melody, prefer 3rd or 5th below ---
        alto_pc_options = [pc for pc in chord_pcs if pc != melody_pc] or chord_pcs
        alto_midi = _closest_pitch_in_range(alto_pc_options, prev_alto_midi, VOICE_RANGES["alto"])

        # ensure alto stays below (or equal to) soprano
        while alto_midi > melody_midi and alto_midi - 12 >= VOICE_RANGES["alto"][0]:
            alto_midi -= 12

        alto.append({
            "start": start, "end": end,
            "midi": alto_midi,
            "pitch_name": librosa.midi_to_note(alto_midi),
        })
        prev_alto_midi = alto_midi

        # --- Tenor: pick a chord tone, generally the root or a 3rd/5th below alto ---
        tenor_pc_options = chord_pcs  # tenor often takes the root
        tenor_midi = _closest_pitch_in_range(tenor_pc_options, prev_tenor_midi, VOICE_RANGES["tenor"])

        # ensure tenor stays below (or equal to) alto
        while tenor_midi > alto_midi and tenor_midi - 12 >= VOICE_RANGES["tenor"][0]:
            tenor_midi -= 12

        tenor.append({
            "start": start, "end": end,
            "midi": tenor_midi,
            "pitch_name": librosa.midi_to_note(tenor_midi),
        })
        prev_tenor_midi = tenor_midi

    return {
        "soprano": soprano,
        "alto": alto,
        "tenor": tenor,
    }
