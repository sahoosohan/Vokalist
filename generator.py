from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import time
from pathlib import Path

from utils import resolve_device
from splitter import SceneChunk, read_script, split_script, title_from_path, write_chunks


DEFAULT_VOICE_REFERENCE = Path("voice_profile/reference.wav")


# NOTE: ChatterboxTurboTTS.generate() accepts `cfg_weight`, `min_p`, and
# `exaggeration` in its signature (inherited for compatibility with the
# non-Turbo ChatterboxTTS) but the Turbo model does not implement them --
# passing them does nothing except trigger a
# "CFG, min_p and exaggeration are not supported by Turbo version" warning
# on every single generation call. So they're deliberately left out of
# these presets. The knobs that actually affect Turbo output are
# temperature, top_p, top_k, and repetition_penalty.
#
# `seed_num` support varies by installed chatterbox-tts version -- some
# builds don't accept it at all (TypeError), which _supported_kwargs()
# below detects and drops automatically.
#
# "vary_seed": True means a fresh random seed is used per generated line
# instead of reusing seed_num=0 for everything -- reusing one seed is part
# of what makes long narration sound flat/robotic, since the model makes
# the exact same micro-decisions every time. Set vary_seed False (or pass
# --seed on the CLI) if you want reproducible output instead of natural
# variation.
STYLE_PRESETS = {
    "neutral": {
        "repetition_penalty": 1.25,
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 50,
        "norm_loudness": True,
        "vary_seed": True,
        "max_generation_words": 85,
        "segment_gap_ms": 120,
        "segment_gap_jitter_ms": 15,
    },
    "storyteller": {
        "repetition_penalty": 1.28,
        "temperature": 0.62,
        "top_p": 0.9,
        "top_k": 50,
        "norm_loudness": True,
        "vary_seed": True,
        "max_generation_words": 70,
        "segment_gap_ms": 130,
        "segment_gap_jitter_ms": 20,
    },
    "storyteller_energetic": {
        "repetition_penalty": 1.22,
        "temperature": 0.68,
        "top_p": 0.92,
        "top_k": 60,
        "norm_loudness": True,
        "vary_seed": True,
        "max_generation_words": 75,
        "segment_gap_ms": 90,
        "segment_gap_jitter_ms": 20,
    },
    "storyteller_excited": {
        # Wider sampling = more vocal variety line-to-line. This is the
        # main lever available on Turbo, since exaggeration/cfg_weight
        # are no-ops (see note above).
        "repetition_penalty": 1.15,
        "temperature": 0.82,
        "top_p": 0.97,
        "top_k": 90,
        "norm_loudness": True,
        "vary_seed": True,
        "max_generation_words": 65,
        "segment_gap_ms": 70,
        "segment_gap_jitter_ms": 25,
    },
}


def generate_audio(
    chunks: list[SceneChunk],
    output_dir: Path,
    voice_reference: Path | None = None,
    device: str = "auto",
    generation_params: dict[str, object] | None = None,
    vary_seed: bool = True,
    fixed_seed: int | None = None,
    max_generation_words: int = 85,
    segment_gap_ms: int = 120,
    segment_gap_jitter_ms: int = 0,
    style: str = "neutral",
    force: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    manifest_path = output_dir / "manifest.json"
    metadata: list[dict[str, object]] = _load_manifest(manifest_path)
    done_stems = {entry["audio_file"] for entry in metadata} if not force else set()

    pending = [c for c in chunks if f"{c.filename_stem}.wav" not in done_stems]
    audio_paths: list[Path] = [output_dir / f"{c.filename_stem}.wav" for c in chunks]

    if not pending:
        print(f"All {len(chunks)} scenes already generated in {output_dir} (pass force=True to regenerate).")
        return audio_paths

    if len(pending) < len(chunks):
        print(f"Resuming: {len(chunks) - len(pending)} scene(s) already done, {len(pending)} to go.")

    model = _load_model(resolved_device)

    # Different chatterbox-tts releases have shipped different generate()
    # signatures. Rather than hardcode a param list that breaks on the next
    # version bump, introspect the installed model and only pass what it
    # actually accepts.
    base_kwargs = dict(generation_params or {})
    base_kwargs.setdefault("seed_num", fixed_seed if fixed_seed is not None else 0)
    generate_kwargs = _supported_kwargs(model.generate, base_kwargs)

    should_vary_seed = vary_seed and fixed_seed is None and "seed_num" in generate_kwargs
    if vary_seed and fixed_seed is None and "seed_num" not in generate_kwargs:
        print(
            "Note: this chatterbox-tts build's generate() has no seed_num, so per-line "
            "variation is coming from temperature/top_p/top_k sampling only."
        )

    for chunk in pending:
        audio_path = output_dir / f"{chunk.filename_stem}.wav"
        started = time.time()
        text_segments = split_for_generation(chunk.text, max_words=max_generation_words)
        wav_segments = []
        seeds_used: list[int] = []

        for segment_index, text_segment in enumerate(text_segments, start=1):
            call_kwargs = dict(generate_kwargs)
            if should_vary_seed:
                call_kwargs["seed_num"] = random.randint(1, 2**31 - 1)
            if "seed_num" in call_kwargs:
                seeds_used.append(call_kwargs["seed_num"])

            wav = model.generate(
                text=prepare_tts_text(text_segment, style=style),
                audio_prompt_path=str(voice_reference) if voice_reference else None,
                **call_kwargs,
            )
            wav_segments.append(_ensure_channels_first(wav).cpu())
            print(f"Generated {chunk.filename_stem} part {segment_index}/{len(text_segments)}")

        stitched_wav = _stitch_waveforms(
            wav_segments, sample_rate=model.sr, gap_ms=segment_gap_ms, gap_jitter_ms=segment_gap_jitter_ms
        )
        _save_wav(audio_path, stitched_wav, model.sr)

        metadata = [entry for entry in metadata if entry["index"] != chunk.index]
        entry = {
            "index": chunk.index,
            "title": chunk.title,
            "text_file": f"chunks/{chunk.filename_stem}.txt",
            "audio_file": audio_path.name,
            "words": len(chunk.text.split()),
            "generation_parts": len(text_segments),
            "style": style,
            "seconds_to_generate": round(time.time() - started, 2),
        }
        if seeds_used:
            entry["seeds"] = seeds_used
        metadata.append(entry)
        metadata.sort(key=lambda entry: entry["index"])
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Generated {audio_path}")

    return audio_paths


def generate_from_script(
    script_path: Path,
    output_dir: Path | None = None,
    voice_reference: Path | None = None,
    max_words: int = 300,
    device: str = "auto",
    max_generation_words: int | None = None,
    style: str = "neutral",
    force: bool = False,
    fixed_seed: int | None = None,
) -> Path:
    style_settings = get_style_settings(style)
    routing_keys = {"max_generation_words", "segment_gap_ms", "segment_gap_jitter_ms", "vary_seed"}
    generation_params = {key: value for key, value in style_settings.items() if key not in routing_keys}

    video_dir = output_dir or Path("output") / title_from_path(script_path)
    chunks = split_script(read_script(script_path), max_words=max_words)
    if not chunks:
        raise ValueError(f"No narration text found in {script_path}. Check the script format.")
    write_chunks(chunks, video_dir / "chunks")
    generate_audio(
        chunks,
        video_dir,
        voice_reference=voice_reference,
        device=device,
        generation_params=generation_params,
        vary_seed=style_settings.get("vary_seed", True),
        fixed_seed=fixed_seed,
        max_generation_words=max_generation_words or style_settings["max_generation_words"],
        segment_gap_ms=style_settings["segment_gap_ms"],
        segment_gap_jitter_ms=style_settings.get("segment_gap_jitter_ms", 0),
        style=style,
        force=force,
    )
    return video_dir


def get_style_settings(style: str) -> dict[str, float | int | bool]:
    if style not in STYLE_PRESETS:
        choices = ", ".join(sorted(STYLE_PRESETS))
        raise ValueError(f"Unknown style '{style}'. Choose one of: {choices}")
    return STYLE_PRESETS[style]


def resolve_voice_reference(voice_reference: Path | None) -> Path:
    reference = voice_reference or DEFAULT_VOICE_REFERENCE
    if not reference.exists():
        raise FileNotFoundError(
            f"Voice reference not found: {reference}. "
            "Place a clean 5-15 second reference clip at voice_profile/reference.wav "
            "or pass --voice-reference with another file."
        )
    return reference


def _load_model(device: str):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device=device)


def _supported_kwargs(func, candidate_kwargs: dict[str, object]) -> dict[str, object]:
    """Return only the entries of candidate_kwargs that func's real signature
    accepts, so a style preset with e.g. seed_num doesn't crash on a
    chatterbox-tts build whose generate() doesn't have that param.
    If func takes **kwargs, everything is passed through untouched."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(candidate_kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(candidate_kwargs)

    supported = {name: value for name, value in candidate_kwargs.items() if name in params}
    dropped = sorted(set(candidate_kwargs) - set(supported))
    if dropped:
        print(
            f"Note: this chatterbox-tts build's generate() doesn't accept {dropped}; "
            "skipping those and using its defaults instead."
        )
    return supported


def _load_manifest(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_wav(path: Path, waveform, sample_rate: int) -> None:
    import torchaudio as ta

    ta.save(str(path), waveform, sample_rate)


def _ensure_channels_first(wav):
    if wav.ndim == 1:
        return wav.unsqueeze(0)
    if wav.ndim == 2:
        return wav
    raise ValueError(f"Expected 1D or 2D waveform tensor, got shape {tuple(wav.shape)}")


def split_for_generation(text: str, max_words: int = 85) -> list[str]:
    if len(text.split()) <= max_words:
        return [text.strip()]

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if current and current_words + len(sentence_words) > max_words:
            segments.append(" ".join(current).strip())
            current = []
            current_words = 0

        if len(sentence_words) > max_words:
            segments.extend(_split_words(sentence_words, max_words))
            continue

        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        segments.append(" ".join(current).strip())

    return segments


def prepare_tts_text(text: str, style: str = "neutral") -> str:
    text = normalize_tts_text(text)
    if style in {"storyteller", "storyteller_energetic", "storyteller_excited"}:
        text = shape_storyteller_pauses(text)
    if style in {"storyteller_energetic", "storyteller_excited"}:
        text = shape_energetic_storyteller_pauses(text)
    if style == "storyteller_excited":
        text = shape_excited_pauses(text)
    return text


def normalize_tts_text(text: str) -> str:
    replacements = {
        "\u2014": ", ",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def shape_storyteller_pauses(text: str) -> str:
    text = re.sub(r"\b(But|So|Now)\b", r"\1,", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\.\s+", ". ", text)
    text = re.sub(r"\?\s+", "? ", text)
    text = re.sub(r"!\s+", "! ", text)
    return text.strip()


def shape_energetic_storyteller_pauses(text: str) -> str:
    text = re.sub(r"\b(And|But|So|Now),\s+", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Sentences that already contain emphatic language but end in a flat period
# read as unnaturally deadpan out loud. This nudges punctuation to match
# content the writer already chose to emphasize -- it never adds words or
# changes meaning, just lets the model's exclamation-point prosody kick in
# where the text was already excited.
_EXCITEMENT_CUES = (
    "wow", "unbelievable", "incredible", "insane", "wild", "crazy",
    "no way", "imagine that", "mind-blowing", "shocking", "astonishing",
    "amazing", "turns out",
)


def shape_excited_pauses(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    reshaped: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped.endswith(".") and any(cue in stripped.lower() for cue in _EXCITEMENT_CUES):
            stripped = stripped[:-1] + "!"
        reshaped.append(stripped)
    return " ".join(reshaped).strip()


def _split_words(words: list[str], max_words: int) -> list[str]:
    return [" ".join(words[index : index + max_words]).strip() for index in range(0, len(words), max_words)]


def _stitch_waveforms(waveforms, sample_rate: int, gap_ms: int, gap_jitter_ms: int = 0):
    import torch

    if not waveforms:
        raise ValueError("No generated waveform segments to stitch.")

    parts = []
    dtype = waveforms[0].dtype
    channels = waveforms[0].shape[0]

    for index, waveform in enumerate(waveforms):
        parts.append(waveform)
        if index + 1 < len(waveforms):
            this_gap_ms = gap_ms
            if gap_jitter_ms:
                this_gap_ms = max(0, gap_ms + random.randint(-gap_jitter_ms, gap_jitter_ms))
            gap_samples = int(sample_rate * this_gap_ms / 1000)
            if gap_samples > 0:
                parts.append(torch.zeros((channels, gap_samples), dtype=dtype))

    return torch.cat(parts, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene WAV files with Chatterbox Turbo.")
    parser.add_argument("script", type=Path, help="Path to a .md or .txt script.")
    parser.add_argument(
        "--voice-reference",
        type=Path,
        default=None,
        help="Clean 5-15 second reference WAV/MP3. Defaults to voice_profile/reference.wav.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for this video.")
    parser.add_argument("--max-words", type=int, default=300, help="Maximum words per TTS chunk.")
    parser.add_argument(
        "--max-generation-words",
        type=int,
        default=None,
        help="Maximum words per individual model generation inside each scene. Defaults depend on --style.",
    )
    parser.add_argument("--style", default="neutral", choices=sorted(STYLE_PRESETS), help="Narration style preset.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="TTS device.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate all scenes even if audio already exists in the output dir."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix a specific seed for reproducible output. Omit for natural per-line variation (default).",
    )
    args = parser.parse_args()

    try:
        voice_reference = resolve_voice_reference(args.voice_reference)
    except FileNotFoundError as exc:
        parser.exit(1, f"Error: {exc}\n")

    try:
        output_dir = generate_from_script(
            script_path=args.script,
            output_dir=args.output_dir,
            voice_reference=voice_reference,
            max_words=args.max_words,
            device=args.device,
            max_generation_words=args.max_generation_words,
            style=args.style,
            force=args.force,
            fixed_seed=args.seed,
        )
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Done. Audio files are in {output_dir}")


if __name__ == "__main__":
    main()