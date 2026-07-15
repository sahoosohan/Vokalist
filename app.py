from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from generator import DEFAULT_VOICE_REFERENCE, STYLE_PRESETS, generate_from_script, resolve_voice_reference
from stitcher import stitch_wavs


def run_pipeline(
    script_text: str,
    script_file,
    voice_reference,
    video_title: str,
    max_words: int,
    style: str,
    gap_ms: int,
    gap_jitter_ms: int,
    device: str,
    force: bool,
    seed: float | None,
):
    if not script_text.strip() and script_file is None:
        raise gr.Error("Paste a script or upload a .md/.txt file.")

    title = _safe_title(video_title) or "video"
    output_dir = Path("output") / title
    script_path = _prepare_script(script_text, script_file, output_dir)
    try:
        voice_path = resolve_voice_reference(Path(voice_reference) if voice_reference else None)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    try:
        generate_from_script(
            script_path=script_path,
            output_dir=output_dir,
            voice_reference=voice_path,
            max_words=int(max_words),
            device=device,
            style=style,
            force=force,
            fixed_seed=int(seed) if seed not in (None, "") else None,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    final_path = stitch_wavs(output_dir, gap_ms=int(gap_ms), gap_jitter_ms=int(gap_jitter_ms))
    scene_files = _ordered_scene_files(output_dir, exclude_name=final_path.name)
    return str(final_path), str(final_path), scene_files, str(output_dir)


def _prepare_script(script_text: str, script_file, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if script_file is not None:
        source = Path(script_file)
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise gr.Error(
                f"Couldn't read {source.name} as UTF-8 text. Re-save it as plain UTF-8 .md/.txt and try again."
            ) from exc
        target = output_dir / source.name
        target.write_text(content, encoding="utf-8")
        return target

    script_path = output_dir / "script.md"
    script_path.write_text(script_text.strip() + "\n", encoding="utf-8")
    return script_path


def _safe_title(title: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in title).strip("_")
    return slug


def _ordered_scene_files(output_dir: Path, exclude_name: str) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries.sort(key=lambda entry: entry["index"])
            paths = [output_dir / entry["audio_file"] for entry in entries]
            if all(p.exists() for p in paths):
                return [str(p) for p in paths if p.name != exclude_name]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return sorted(str(path) for path in output_dir.glob("*.wav") if path.name != exclude_name)


with gr.Blocks(title="Vokalist") as demo:
    gr.Markdown("# Vokalist")
    with gr.Row():
        with gr.Column(scale=2):
            script_text = gr.Textbox(label="Script", lines=18, placeholder="Paste your scene-by-scene script here.")
            script_file = gr.File(label="Or upload .md/.txt script", file_types=[".md", ".txt"], type="filepath")
            voice_reference = gr.Audio(
                label=f"Voice reference (defaults to {DEFAULT_VOICE_REFERENCE})",
                type="filepath",
            )
        with gr.Column(scale=1):
            video_title = gr.Textbox(
                label="Video title",
                value="why_behind_us",
                info="Used as the output folder name (letters/numbers only). Reusing a title resumes that video's progress unless Force regenerate is checked.",
            )
            max_words = gr.Slider(label="Max words per chunk", minimum=120, maximum=500, value=300, step=10)
            style = gr.Radio(
                label="Style",
                choices=sorted(STYLE_PRESETS),
                value="storyteller_energetic",
                info="storyteller_excited adds more vocal variety and tighter pacing for a more human, energetic read.",
            )
            gap_ms = gr.Slider(label="Scene gap (ms)", minimum=0, maximum=1500, value=450, step=50)
            gap_jitter_ms = gr.Slider(
                label="Scene gap jitter (ms)",
                minimum=0,
                maximum=300,
                value=0,
                step=10,
                info="Randomly varies each scene gap by up to this much so pacing isn't perfectly uniform.",
            )
            device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu", "mps"], value="auto")
            force = gr.Checkbox(
                label="Force regenerate all scenes",
                value=False,
                info="Off = skip scenes already generated for this title (resume). On = regenerate everything.",
            )
            seed = gr.Number(
                label="Fixed seed (optional)",
                value=None,
                precision=0,
                info="Leave blank for natural per-line variation (recommended). Set a value for reproducible output.",
            )
            run_button = gr.Button("Generate Narration", variant="primary")

    final_audio = gr.Audio(label="Final narration", type="filepath")
    final_download = gr.File(label="Download final narration")
    scene_files = gr.Files(label="Scene WAV files")
    output_dir = gr.Textbox(label="Output folder")

    run_button.click(
        run_pipeline,
        inputs=[
            script_text,
            script_file,
            voice_reference,
            video_title,
            max_words,
            style,
            gap_ms,
            gap_jitter_ms,
            device,
            force,
            seed,
        ],
        outputs=[final_audio, final_download, scene_files, output_dir],
    )


if __name__ == "__main__":
    server_name = os.getenv("VOKALIST_HOST", "127.0.0.1")
    server_port = int(os.getenv("VOKALIST_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)