"""
Lyrics Transcription & Alignment Module
==========================================
Uses OpenAI's Whisper model to transcribe the vocal track and produce
word-level (or segment-level) timestamps, so lyrics can be displayed
in sync with playback - highlighting the current word/line as each
voice part plays.

NOTE ON DEPLOYMENT
------------------
Whisper models range from "tiny" (~75MB, fast, less accurate) to
"large" (~3GB, slow, most accurate). For a responsive web app,
"base" or "small" are good defaults. Like Demucs, this should run
as a background job in production, not inline in a request/response
cycle, since transcription of a 3-4 minute song can take 10-60+
seconds on CPU.
"""

import whisper

_model_cache = {}


def _get_model(model_size: str = "base"):
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]


def transcribe_with_timing(audio_path: str, model_size: str = "base"):
    """
    Transcribe an audio file and return lyrics with timing.

    Returns
    -------
    dict with:
      - text: full transcript
      - segments: list of {start, end, text} - line-level timing
      - words: list of {start, end, word} - word-level timing
               (only if the whisper version supports word_timestamps)
    """
    model = _get_model(model_size)

    result = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    words = []

    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

        for w in seg.get("words", []):
            words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"].strip(),
            })

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "words": words,
        "language": result.get("language"),
    }


def align_lyrics_to_voice_part(words, voice_notes):
    """
    For a given voice part's note sequence, attach the corresponding
    lyric word(s) that fall within each note's time span. This lets
    the UI display "which word goes with which note" for that voice.

    Parameters
    ----------
    words : list of {start, end, word} from transcribe_with_timing
    voice_notes : list of {start, end, midi, pitch_name}

    Returns
    -------
    The voice_notes list, with an added "lyric" field per note
    (empty string if no word overlaps that note's time span).
    """
    annotated = []
    for note in voice_notes:
        note_copy = dict(note)
        overlapping = [
            w["word"] for w in words
            if w["start"] < note["end"] and w["end"] > note["start"]
        ]
        note_copy["lyric"] = " ".join(overlapping)
        annotated.append(note_copy)

    return annotated
