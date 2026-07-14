from __future__ import annotations

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
    device: str,
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

    generate_from_script(
        script_path=script_path,
        output_dir=output_dir,
        voice_reference=voice_path,
        max_words=max_words,
        device=device,
        style=style,
    )
    final_path = stitch_wavs(output_dir, gap_ms=gap_ms)
    scene_files = sorted(str(path) for path in output_dir.glob("*.wav") if path.name != final_path.name)
    return str(final_path), str(final_path), scene_files, str(output_dir)


def _prepare_script(script_text: str, script_file, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if script_file is not None:
        source = Path(script_file)
        target = output_dir / source.name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    script_path = output_dir / "script.md"
    script_path.write_text(script_text.strip() + "\n", encoding="utf-8")
    return script_path


def _safe_title(title: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in title).strip("_")


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
            video_title = gr.Textbox(label="Video title", value="why_behind_us")
            max_words = gr.Slider(label="Max words per chunk", minimum=120, maximum=500, value=300, step=10)
            style = gr.Radio(label="Style", choices=sorted(STYLE_PRESETS), value="storyteller_energetic")
            gap_ms = gr.Slider(label="Scene gap (ms)", minimum=0, maximum=1500, value=450, step=50)
            device = gr.Radio(label="Device", choices=["auto", "cuda", "cpu"], value="auto")
            run_button = gr.Button("Generate Narration", variant="primary")

    final_audio = gr.Audio(label="Final narration", type="filepath")
    final_download = gr.File(label="Download final narration")
    scene_files = gr.Files(label="Scene WAV files")
    output_dir = gr.Textbox(label="Output folder")

    run_button.click(
        run_pipeline,
        inputs=[script_text, script_file, voice_reference, video_title, max_words, style, gap_ms, device],
        outputs=[final_audio, final_download, scene_files, output_dir],
    )


if __name__ == "__main__":
    server_name = os.getenv("VOKALIST_HOST", "127.0.0.1")
    server_port = int(os.getenv("VOKALIST_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
