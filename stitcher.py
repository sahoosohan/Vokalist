from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torchaudio as ta


def stitch_wavs(
    input_dir: Path, output_path: Path | None = None, gap_ms: int = 400, gap_jitter_ms: int = 0
) -> Path:
    output_path = output_path or input_dir / "final_narration.wav"
    wav_paths = _ordered_wav_paths(input_dir, exclude_name=output_path.name)

    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    combined_parts: list[torch.Tensor] = []
    target_sr: int | None = None
    target_channels: int | None = None
    trailing_gap_appended = False

    for wav_path in wav_paths:
        waveform, sample_rate = ta.load(str(wav_path))
        if target_sr is None:
            target_sr = sample_rate
            target_channels = waveform.shape[0]
        elif sample_rate != target_sr:
            waveform = ta.functional.resample(waveform, sample_rate, target_sr)

        waveform = _match_channels(waveform, target_channels or waveform.shape[0])
        combined_parts.append(waveform)
        trailing_gap_appended = False

        this_gap_ms = gap_ms
        if gap_jitter_ms:
            this_gap_ms = max(0, gap_ms + random.randint(-gap_jitter_ms, gap_jitter_ms))
        gap_samples = int((target_sr or sample_rate) * this_gap_ms / 1000)
        if gap_samples > 0:
            combined_parts.append(torch.zeros((waveform.shape[0], gap_samples), dtype=waveform.dtype))
            trailing_gap_appended = True

    if trailing_gap_appended:
        combined_parts.pop()

    if target_sr is None:
        raise FileNotFoundError(f"No stitchable .wav files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), torch.cat(combined_parts, dim=1), target_sr)
    return output_path


def _ordered_wav_paths(input_dir: Path, exclude_name: str) -> list[Path]:
    """Prefer manifest.json's `index` order (written by generator.py) since it's
    an explicit source of truth. Fall back to lexical filename sort if there's
    no manifest, so this still works on a bare folder of numbered WAVs."""
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries.sort(key=lambda entry: entry["index"])
            paths = [input_dir / entry["audio_file"] for entry in entries]
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"Warning: manifest lists {len(missing)} missing file(s), falling back to directory scan.")
            else:
                return [p for p in paths if p.name != exclude_name]
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    return sorted(p for p in input_dir.glob("*.wav") if p.name != exclude_name)


def _match_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    if waveform.shape[0] == channels:
        return waveform
    if waveform.shape[0] == 1 and channels == 2:
        return waveform.repeat(2, 1)
    if waveform.shape[0] == 2 and channels == 1:
        return waveform.mean(dim=0, keepdim=True)
    raise ValueError(f"Cannot convert {waveform.shape[0]} channels to {channels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch numbered scene WAV files into one narration WAV.")
    parser.add_argument("input_dir", type=Path, help="Directory containing numbered .wav files.")
    parser.add_argument("--output", type=Path, default=None, help="Final output path.")
    parser.add_argument("--gap-ms", type=int, default=400, help="Silence gap between scenes in milliseconds.")
    parser.add_argument(
        "--gap-jitter-ms",
        type=int,
        default=0,
        help="Randomly vary each inter-scene gap by up to this many ms so pacing isn't perfectly uniform.",
    )
    args = parser.parse_args()

    output_path = stitch_wavs(
        args.input_dir, output_path=args.output, gap_ms=args.gap_ms, gap_jitter_ms=args.gap_jitter_ms
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()