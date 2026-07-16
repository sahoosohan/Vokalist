# Vokalist

Vokalist is a local AI narration tool for scene-based video scripts. It splits a Markdown or text script into numbered narration sections, generates cloned-voice WAV files with Chatterbox Turbo, and stitches them into a final narration track.

## What It Does

- Parses scene scripts such as Hook, Setup, Segment 1, Payoff, and CTA.
- Removes non-spoken script metadata such as headings, timestamps, separators, word count, and runtime notes.
- Uses `voice_profile/reference.wav` as the default voice reference.
- Generates numbered scene WAV files.
- Stitches all scenes into `final_narration.wav`, ordered by `manifest.json` (falls back to filename sort if the manifest is missing).
- Resumes automatically: if you rerun the same output folder, already-generated scenes are skipped. Pass `--force` (CLI) or check "Force regenerate" (UI) to redo everything.
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
├── generator.py           # Chatterbox Turbo generation and style presets
├── splitter.py            # Script parsing and narration cleanup
├── stitcher.py            # WAV stitching
├── utils.py               # Shared device resolution + slug helper
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

## Resuming / Regenerating

Reruns of `pipeline.py` or the UI against the *same* output folder skip any scene that already has a WAV file in `manifest.json` — handy if generation crashes or you kill it partway through. To force a clean regenerate of every scene:

```powershell
.\.venv\Scripts\python.exe pipeline.py scripts\my_script.md --output-dir output\my_script --force
```

## Style Presets

- `neutral`: stable, plain narration.
- `storyteller`: warmer and slower, with more deliberate pacing.
- `storyteller_energetic`: clear but less lazy, with tighter pauses and more forward motion.
- `storyteller_excited`: widest sampling variety and tightest pacing, for a more human, energetic read. Also nudges punctuation to `!` on sentences that already contain emphatic language ("incredible", "no way", etc.) — it never adds words, just lets existing emphasis come through in the delivery.

For your current workflow, use:

```text
storyteller_energetic
```

### Making narration sound more human

A few things beyond style choice help avoid a flat, robotic read:

- **Per-line seed variation** (on by default, all presets): each generated line uses a fresh random seed instead of reusing the same one for the whole video. Reusing one seed is a big part of what makes long narration sound identical/robotic line to line. Pass `--seed N` (CLI) or set "Fixed seed" (UI) if you want reproducible output instead — useful for A/B comparing a specific line, at the cost of that flat sameness.
- **Scene gap jitter**: `--gap-jitter-ms` (CLI) / "Scene gap jitter" (UI) randomly varies each pause by up to that many ms so the pacing isn't metronomically uniform. Try 15-30ms.
- **Wider sampling on `storyteller_excited`**: higher `temperature`/`top_p`/`top_k` and lower `repetition_penalty` than the other presets — this is the main lever available for expressiveness on Turbo. (`exaggeration`/`cfg_weight` are *not* used: Turbo accepts those kwargs but ignores them entirely — see Troubleshooting.)

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
- If generation is too slow, confirm `--device cuda` and `torch.cuda.is_available()`. On Apple Silicon, `--device mps` is also supported.
- If the app says the reference is missing, add `voice_profile/reference.wav`.
- If a run gets interrupted, just rerun the same command — it resumes from the last completed scene. Use `--force` to start over.
- `ChatterboxTurboTTS.generate()`'s accepted parameters vary by installed `chatterbox-tts` version (some builds don't support `seed_num` at all and raise `TypeError`). `generator.py` inspects the installed model's real signature at runtime and only passes params it actually accepts, printing a `Note: ... doesn't accept [...]` line for anything it had to drop — that's informational, not an error.
- If you see `WARNING:chatterbox.tts_turbo:CFG, min_p and exaggeration are not supported by Turbo version and will be ignored` — that's the library itself, not this codebase. Those three params are accepted by Turbo's `generate()` signature but do nothing; `STYLE_PRESETS` no longer sets them, so you shouldn't see this warning anymore. If you do, you're likely on an older cached copy of `generator.py`.