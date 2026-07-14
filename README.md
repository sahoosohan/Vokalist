# Vokalist

Vokalist is a local AI narration tool for scene-based video scripts. It splits a Markdown or text script into numbered narration sections, generates cloned-voice WAV files with Chatterbox Turbo, and stitches them into a final narration track.

## What It Does

- Parses scene scripts such as Hook, Setup, Segment 1, Payoff, and CTA.
- Removes non-spoken script metadata such as headings, timestamps, separators, word count, and runtime notes.
- Uses `voice_profile/reference.wav` as the default voice reference.
- Generates numbered scene WAV files.
- Stitches all scenes into `final_narration.wav`.
- Provides a Gradio UI with audio preview, final narration download, scene file downloads, and output folder display.

## Requirements

- Windows or Linux host.
- Python 3.10+.
- NVIDIA GPU strongly recommended.
- CUDA-capable PyTorch environment for practical generation speed.
- 8 GB VRAM recommended for Chatterbox Turbo.
- A clean 5-15 second reference recording at:

```text
voice_profile/reference.wav
```

## Project Structure

```text
Vokalist/
├── app.py                 # Gradio UI
├── pipeline.py            # Full split -> generate -> stitch CLI
├── generator.py           # Chatterbox generation and style presets
├── splitter.py            # Script parsing and narration cleanup
├── stitcher.py            # WAV stitching
├── requirements.txt       # Python dependencies
├── scripts/               # Input scripts
├── voice_profile/         # Voice reference audio
└── output/                # Generated chunks, scene WAVs, final WAVs
```

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm CUDA is available:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Run the UI

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:7860
```

Recommended UI settings:

```text
Max words per chunk: 300
Style: storyteller_energetic
Scene gap: 450 ms
Device: cuda
```

After generation, use `Download final narration` to save `final_narration.wav` from the browser.

## Run from CLI

```powershell
.\.venv\Scripts\python.exe pipeline.py scripts\The_People_Who_Cant_Picture_Anything.md --voice-reference voice_profile\reference.wav --output-dir output\the_people_who_cant_picture_anything --device cuda --gap-ms 450 --style storyteller_energetic
```

Generated output:

```text
output\[video_title]\
├── chunks\*.txt
├── 01_*.wav
├── 02_*.wav
├── ...
├── final_narration.wav
└── manifest.json
```

## Style Presets

- `neutral`: stable, plain narration.
- `storyteller`: warmer and slower, with more deliberate pacing.
- `storyteller_energetic`: clear but less lazy, with tighter pauses and more forward motion.

For your current workflow, use:

```text
storyteller_energetic
```

## Deployment Notes

- The app listens on `127.0.0.1:7860` locally.
- You can override host and port with:

```powershell
$env:VOKALIST_HOST="0.0.0.0"
$env:VOKALIST_PORT="7860"
.\.venv\Scripts\python.exe app.py
```

- Keep `voice_profile/reference.wav` present before generating.
- Do not expose this app publicly without authentication, because it can run expensive local GPU jobs and read/write local folders.

## Script Format

Use Markdown headings or plain labels:

```markdown
### 1. THE HOOK (0:00-0:15)
Narration text...

#### Segment 1 - Main idea (0:45-2:30)
Narration text...

### TAKEAWAY + CTA
Narration text...
```

The splitter uses headings for chunk names, but only sends narration body text to TTS.

## Troubleshooting

- If the voice changes or becomes gibberish, use `storyteller_energetic` or lower `--max-generation-words`.
- If generation is too slow, confirm `--device cuda` and `torch.cuda.is_available()`.
- If the app says the reference is missing, add `voice_profile/reference.wav`.