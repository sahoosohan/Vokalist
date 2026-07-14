from chatterbox.tts_turbo import ChatterboxTurboTTS
import torchaudio as ta
import torch

print("CUDA available:", torch.cuda.is_available())

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

text = "Chatterbox Turbo is fast, expressive, and open source."
wav = model.generate(text=text)

ta.save("output.wav", wav, model.sr)
print("Done — check output.wav")