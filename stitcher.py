from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio as ta


def stitch_wavs(input_dir: Path, output_path: Path | None = None, gap_ms: int = 400) -> Path:
    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    output_path = output_path or input_dir / "final_narration.wav"
    combined_parts: list[torch.Tensor] = []
    target_sr: int | None = None
    target_channels: int | None = None

    for wav_path in wav_paths:
        if wav_path.name == output_path.name:
            continue
        waveform, sample_rate = ta.load(str(wav_path))
        if target_sr is None:
            target_sr = sample_rate
            target_channels = waveform.shape[0]
        elif sample_rate != target_sr:
            waveform = ta.functional.resample(waveform, sample_rate, target_sr)

        waveform = _match_channels(waveform, target_channels or waveform.shape[0])
        combined_parts.append(waveform)

        gap_samples = int((target_sr or sample_rate) * gap_ms / 1000)
        if gap_samples > 0:
            combined_parts.append(torch.zeros((waveform.shape[0], gap_samples), dtype=waveform.dtype))

    if combined_parts and gap_ms > 0:
        combined_parts.pop()

    if target_sr is None:
        raise FileNotFoundError(f"No stitchable .wav files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), torch.cat(combined_parts, dim=1), target_sr)
    return output_path


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
    args = parser.parse_args()

    output_path = stitch_wavs(args.input_dir, output_path=args.output, gap_ms=args.gap_ms)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
