from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to 'cuda', 'mps', or 'cpu'. Kept in one place so every
    entry point (UI, pipeline CLI, generator CLI) makes the same decision
    the same way."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_slug(text: str) -> str:
    """Turn arbitrary user text into a filesystem-safe slug.
    Collapses repeated separators so 'Video!!!' and 'Video???' don't both become
    'video_' - they both become 'video' and are visually indistinguishable, but
    at least neither picks up a stray trailing underscore."""
    slug = "".join(char.lower() if char.isalnum() else "_" for char in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "video"