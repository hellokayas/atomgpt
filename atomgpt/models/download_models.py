#!/usr/bin/env python3
"""
Download specified base models into exact local locations using huggingface_hub.snapshot_download.

Usage:
  python download_base_models.py
  # or (optionally) pass an HF token:
  HF_TOKEN= generate token using this https://huggingface.co/settings/tokens and then use export HF_TOKEN={your_token_here}
"""

import os
import sys
from huggingface_hub import snapshot_download

BASE_MODELS = {
    "gemma": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/gemma-3-12b-it-bnb-4bit",
    "llama": "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/Llama-3.2-11B-Vision-Instruct",
    "qwen":  "/projects/p32726/microscopy-gpt/atomgpt/atomgpt/models/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
}

def ensure_writable_dir(path: str) -> None:
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    # create the target dir as well (snapshot_download will create, but this validates perms early)
    os.makedirs(path, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise PermissionError(f"Target directory is not writable: {path}")

def download_model(model_id: str, local_dir: str, token: str | None) -> None:
    print(f"\n==> Downloading: {model_id}")
    print(f"    To: {local_dir}")
    ensure_writable_dir(local_dir)

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # store real files (safer on shared filesystems)
        token=token,                   # supports private/gated repos if provided
        resume_download=True,
    )
    print(f"    Done: {model_id}")

def main() -> int:
    # If your repos are gated/private, set HF_TOKEN env var.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    failures: list[tuple[str, str]] = []
    for name, path in BASE_MODELS.items():
        model_id = f"unsloth/{os.path.basename(path)}"
        try:
            download_model(model_id=model_id, local_dir=path, token=token)
        except Exception as e:
            failures.append((model_id, str(e)))
            print(f"    ERROR downloading {model_id}: {e}", file=sys.stderr)

    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for mid, err in failures:
            print(f"  - {mid}: {err}", file=sys.stderr)
        return 1

    print("\nAll downloads completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
