"""
Audio Synthesis Module
========================
Renders a sequence of note events (start, end, midi) into an audible
WAV file using simple additive synthesis (sine waves + harmonics +
envelope shaping). This produces a clear, pitch-accurate "reference
tone" track for each voice part - similar to how choir rehearsal apps
(e.g. practice-track generators) render parts before a human ever
sings them.

This does NOT attempt realistic vocal timbre/singing-voice synthesis
(that requires GPU-hosted neural vocoders such as DiffSinger or
so-vits-svc). What it provides is accurate, audible pitches that choir
members can use to learn their part - a "practice piano/oboe" tone.

Each voice gets a slightly different timbre (different harmonic mix)
so Soprano/Alto/Tenor are easy to distinguish by ear.
"""

import numpy as np
from scipy.io import wavfile

SR = 44100  # output sample rate


def _adsr_envelope(n_samples, sr, attack=0.02, release=0.08):
    """Simple attack/release envelope to avoid clicks between notes."""
    env = np.ones(n_samples)
    a = int(attack * sr)
    r = int(release * sr)
    a = min(a, n_samples // 2)
    r = min(r, n_samples // 2)

    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if r > 0:
        env[-r:] = np.linspace(1, 0, r)
    return env


# Harmonic recipes per voice - gives each part a distinct, pleasant timbre
VOICE_TIMBRES = {
    "soprano": [(1, 1.0), (2, 0.35), (3, 0.12), (4, 0.05)],
    "alto":    [(1, 1.0), (2, 0.45), (3, 0.20), (4, 0.08)],
    "tenor":   [(1, 1.0), (2, 0.55), (3, 0.28), (4, 0.12)],
}


def midi_to_freq(midi_number):
    return 440.0 * (2 ** ((midi_number - 69) / 12))


def render_voice_part(notes, duration_seconds, voice_name="soprano", sr=SR):
    """
    Render a list of note events into a mono audio buffer.

    Parameters
    ----------
    notes : list of {start, end, midi}
    duration_seconds : float - total length of the output track
    voice_name : "soprano" | "alto" | "tenor" - selects timbre

    Returns
    -------
    np.ndarray (float32), shape (n_samples,), range approx [-1, 1]
    """
    n_total = int(duration_seconds * sr) + sr  # pad 1s for safety
    buffer = np.zeros(n_total, dtype=np.float64)

    harmonics = VOICE_TIMBRES.get(voice_name, VOICE_TIMBRES["soprano"])

    for note in notes:
        start_sample = int(note["start"] * sr)
        end_sample = int(note["end"] * sr)
        n_samples = end_sample - start_sample
        if n_samples <= 0:
            continue

        t = np.arange(n_samples) / sr
        freq = midi_to_freq(note["midi"])

        tone = np.zeros(n_samples)
        for harmonic_num, amplitude in harmonics:
            tone += amplitude * np.sin(2 * np.pi * freq * harmonic_num * t)

        # normalize harmonics sum so peak amplitude stays sane
        tone /= sum(a for _, a in harmonics)

        env = _adsr_envelope(n_samples, sr)
        tone *= env

        end_idx = min(start_sample + n_samples, n_total)
        actual_len = end_idx - start_sample
        buffer[start_sample:end_idx] += tone[:actual_len] * 0.6

    return buffer


def save_wav(buffer: np.ndarray, filepath: str, sr=SR):
    """Save a float audio buffer as a 16-bit PCM WAV file."""
    # normalize to avoid clipping
    peak = np.max(np.abs(buffer)) if len(buffer) else 1.0
    if peak > 1.0:
        buffer = buffer / peak

    int_buffer = (buffer * 32767).astype(np.int16)
    wavfile.write(filepath, sr, int_buffer)


def change_playback_speed(buffer: np.ndarray, speed: float):
    """
    Time-stretch a buffer to a different playback speed WITHOUT
    changing pitch, using librosa's phase-vocoder time stretch.

    speed < 1.0 = slower (e.g. 0.5 = half speed)
    speed > 1.0 = faster
    """
    import librosa
    if speed == 1.0:
        return buffer
    return librosa.effects.time_stretch(buffer.astype(np.float32), rate=speed)
