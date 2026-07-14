from __future__ import annotations

import argparse
from pathlib import Path

from generator import STYLE_PRESETS, generate_from_script, resolve_voice_reference
from stitcher import stitch_wavs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Vokalist script-to-final-WAV pipeline.")
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
    parser.add_argument("--gap-ms", type=int, default=400, help="Silence gap between scenes in milliseconds.")
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
    final_path = stitch_wavs(output_dir, gap_ms=args.gap_ms)
    print(f"Final narration: {final_path}")


if __name__ == "__main__":
    main()
