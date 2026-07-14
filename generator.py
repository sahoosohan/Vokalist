from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from splitter import SceneChunk, read_script, split_script, title_from_path, write_chunks


DEFAULT_VOICE_REFERENCE = Path("voice_profile/reference.wav")


STYLE_PRESETS = {
    "neutral": {
        "repetition_penalty": 1.25,
        "temperature": 0.65,
        "top_p": 0.9,
        "exaggeration": 0.0,
        "cfg_weight": 0.0,
        "max_generation_words": 85,
        "segment_gap_ms": 120,
    },
    "storyteller": {
        "repetition_penalty": 1.28,
        "temperature": 0.62,
        "top_p": 0.9,
        "exaggeration": 0.0,
        "cfg_weight": 0.0,
        "max_generation_words": 70,
        "segment_gap_ms": 130,
    },
    "storyteller_energetic": {
        "repetition_penalty": 1.22,
        "temperature": 0.68,
        "top_p": 0.92,
        "exaggeration": 0.0,
        "cfg_weight": 0.0,
        "max_generation_words": 75,
        "segment_gap_ms": 90,
    },
}


def generate_audio(
    chunks: list[SceneChunk],
    output_dir: Path,
    voice_reference: Path | None = None,
    device: str = "auto",
    repetition_penalty: float = 1.25,
    temperature: float = 0.65,
    top_p: float = 0.9,
    exaggeration: float = 0.0,
    cfg_weight: float = 0.0,
    max_generation_words: int = 85,
    segment_gap_ms: int = 120,
    style: str = "neutral",
) -> list[Path]:
    import torch
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    model = ChatterboxTurboTTS.from_pretrained(device=resolved_device)

    audio_paths: list[Path] = []
    metadata: list[dict[str, object]] = []

    for chunk in chunks:
        audio_path = output_dir / f"{chunk.filename_stem}.wav"
        started = time.time()
        text_segments = split_for_generation(chunk.text, max_words=max_generation_words)
        wav_segments = []

        for segment_index, text_segment in enumerate(text_segments, start=1):
            wav = model.generate(
                text=prepare_tts_text(text_segment, style=style),
                audio_prompt_path=str(voice_reference) if voice_reference else None,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            wav_segments.append(_ensure_channels_first(wav).cpu())
            print(f"Generated {chunk.filename_stem} part {segment_index}/{len(text_segments)}")

        stitched_wav = _stitch_waveforms(wav_segments, sample_rate=model.sr, gap_ms=segment_gap_ms)
        ta.save(str(audio_path), stitched_wav, model.sr)
        audio_paths.append(audio_path)
        metadata.append(
            {
                "index": chunk.index,
                "title": chunk.title,
                "text_file": f"chunks/{chunk.filename_stem}.txt",
                "audio_file": audio_path.name,
                "words": len(chunk.text.split()),
                "generation_parts": len(text_segments),
                "style": style,
                "seconds_to_generate": round(time.time() - started, 2),
            }
        )
        print(f"Generated {audio_path}")

    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return audio_paths


def generate_from_script(
    script_path: Path,
    output_dir: Path | None = None,
    voice_reference: Path | None = None,
    max_words: int = 300,
    device: str = "auto",
    max_generation_words: int | None = None,
    style: str = "neutral",
) -> Path:
    style_settings = get_style_settings(style)
    video_dir = output_dir or Path("output") / title_from_path(script_path)
    chunks = split_script(read_script(script_path), max_words=max_words)
    write_chunks(chunks, video_dir / "chunks")
    generate_audio(
        chunks,
        video_dir,
        voice_reference=voice_reference,
        device=device,
        repetition_penalty=style_settings["repetition_penalty"],
        temperature=style_settings["temperature"],
        top_p=style_settings["top_p"],
        exaggeration=style_settings["exaggeration"],
        cfg_weight=style_settings["cfg_weight"],
        max_generation_words=max_generation_words or style_settings["max_generation_words"],
        segment_gap_ms=style_settings["segment_gap_ms"],
        style=style,
    )
    return video_dir


def get_style_settings(style: str) -> dict[str, float | int]:
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


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    if style in {"storyteller", "storyteller_energetic"}:
        text = shape_storyteller_pauses(text)
    if style == "storyteller_energetic":
        text = shape_energetic_storyteller_pauses(text)
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


def _split_words(words: list[str], max_words: int) -> list[str]:
    return [" ".join(words[index : index + max_words]).strip() for index in range(0, len(words), max_words)]


def _stitch_waveforms(waveforms, sample_rate: int, gap_ms: int):
    import torch

    if not waveforms:
        raise ValueError("No generated waveform segments to stitch.")

    parts = []
    gap_samples = int(sample_rate * gap_ms / 1000)
    gap = torch.zeros((waveforms[0].shape[0], gap_samples), dtype=waveforms[0].dtype)

    for index, waveform in enumerate(waveforms):
        parts.append(waveform)
        if index + 1 < len(waveforms) and gap_samples > 0:
            parts.append(gap)

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
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="TTS device.")
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
        )
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Done. Audio files are in {output_dir}")


if __name__ == "__main__":
    main()
