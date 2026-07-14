# PRD: Local AI Voiceover Tool — "Why Behind Us" Narration Pipeline

## 1. Overview

A local, self-hosted tool that converts video scripts (broken into scene-by-scene chunks) into narration audio using an open-source TTS model, running entirely on your own PC (RTX 4060 8GB, i7 14700K, 16GB RAM).

**Goal:** paste/upload a script → get numbered, ready-to-edit audio files → stitch into one final narration track, with zero per-character API costs and a consistent cloned voice across every video.

---

## 2. Model Choice

**Chatterbox-Turbo** (Resemble AI, MIT license)
- 0.5B parameter Llama-based TTS
- Zero-shot voice cloning from ~5 seconds of reference audio
- Runs comfortably on an 8GB GPU
- Fully free for commercial use

**Fallback if performance issues arise:** Kokoro-82M (Apache 2.0) — lighter, CPU-friendly, no cloning, but a safe backup if Chatterbox runs into VRAM/driver issues.

---

## 3. System Requirements Check

| Requirement | Your Spec | Status |
|---|---|---|
| GPU | RTX 4060 8GB | ✅ Sufficient |
| CUDA support | Needs CUDA 12.x drivers | Verify/install |
| RAM | 16GB | ✅ Sufficient for this workload |
| Python | 3.10+ | Install if not present |
| Disk space | ~5-10GB for model weights + deps | ✅ Fine on 500GB SSD |

---

## 4. Pipeline Architecture

```
[Script (.md/.txt)] 
      ↓
[Script Splitter] → splits into scene chunks (Hook, Setup, Segment 1-5, Payoff, CTA)
      ↓
[Chatterbox-Turbo TTS Engine] → generates audio per chunk, using your cloned voice
      ↓
[Numbered .wav/.mp3 files] → scene_01_hook.mp3, scene_02_setup.mp3, etc.
      ↓
[Stitcher (pydub)] → combines into one continuous narration.mp3 (optional pauses between scenes)
      ↓
[Final Output] → ready to drop into video editor, aligned to your visual cuts
```

---

## 5. Step-by-Step Build Process

### Step 1 — Environment Setup
- Install Python 3.10+ and confirm CUDA 12.x drivers are active (`nvidia-smi` in terminal)
- Create a virtual environment: `python -m venv tts-env`
- Activate it and install PyTorch with CUDA support matching your driver version

- If error shows: "cannot be loaded because running scripts is disabled on this system", run this once: 
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 2 — Install Chatterbox-Turbo
- Install via pip from the official Resemble AI package, or clone the GitHub repo
- Download model weights from Hugging Face (automatic on first run, or manual download)
- Run the official quickstart example to confirm the model loads and produces audio on your GPU

### Step 3 — Create Your Voice Clone
- Record (or select) a clean 5-15 second reference audio clip — ideally a calm, clear narration-style sample with no background noise
- Feed it to Chatterbox's voice cloning function to generate your channel's reusable voice profile
- Save this voice profile so every future video uses the same voice — this becomes your channel's audio identity

### Step 4 — Build the Script Splitter
- Write a small Python script that takes your markdown script file
- Parses it by section headers (Hook, Setup, Segment 1, Segment 2, etc. — matching the format you already use)
- Outputs each section as a separate text chunk, keeping each under ~300 words to stay within safe generation limits

### Step 5 — Build the Generation Loop
- Python script loops through each text chunk
- Sends each to Chatterbox-Turbo with your cloned voice profile
- Saves output as numbered audio files (e.g., `01_hook.mp3`, `02_setup.mp3`) in an output folder named after the video title

### Step 6 — Build the Stitcher
- Use `pydub` to load all numbered audio files in order
- Concatenate them into one file, with a small configurable silence gap (e.g., 400ms) between scenes for natural pacing
- Export as `final_narration.mp3` in the same output folder

### Step 7 — (Optional) Build a Simple Local UI
- Use Gradio to create a lightweight local web interface:
  - Text box to paste script
  - Button: "Generate Scenes"
  - Preview player for each generated chunk
  - Button: "Stitch & Export Final"
- This removes the need to touch code for every new video — just paste and click

### Step 8 — Test on the Aphantasia Script
- Run your existing 12-minute script through the full pipeline
- Check for: pronunciation issues, unnatural pauses, pacing mismatches, and overall voice consistency across scenes
- Adjust chunk boundaries or regenerate specific scenes as needed

### Step 9 — Establish Your Repeatable Workflow
- Once working, this becomes: write script → run splitter → run generator → run stitcher → drop into editor
- Total manual time per video should drop to minutes, not hours

---

## 6. File/Folder Structure (Suggested)

```
tts-tool/
├── voice_profile/           # your cloned voice reference
├── scripts/                 # input .md script files
├── output/
│   └── [video_title]/
│       ├── 01_hook.mp3
│       ├── 02_setup.mp3
│       ├── ...
│       └── final_narration.mp3
├── splitter.py
├── generator.py
├── stitcher.py
└── app.py                   # optional Gradio UI
```

---

## 7. Success Criteria

- [ ] Voice sounds consistent across all scene files (same tone, no jarring pitch shifts)
- [ ] Full 12-minute script processes without manual intervention beyond review
- [ ] Total generation time under 5 minutes per video on your hardware
- [ ] Stitched output has natural pacing with no jarring cuts between scenes
- [ ] Same voice profile reusable across all future videos with zero re-cloning needed

---

## 8. Next Steps After This PRD

1. Confirm CUDA/driver setup is ready
2. I build `splitter.py`, `generator.py`, and `stitcher.py` for you
3. Test end-to-end using the aphantasia script as the first real run
4. Optionally wrap in the Gradio UI once the core pipeline is confirmed working
