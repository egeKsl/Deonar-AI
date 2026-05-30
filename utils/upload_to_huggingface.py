"""
Upload goat detection model weights to HuggingFace Hub.

Usage:
    python utils/upload_to_huggingface.py

Requirements:
    pip install huggingface_hub

Steps before running:
    1. Create account at https://huggingface.co
    2. Go to https://huggingface.co/settings/tokens → New token (write access)
    3. Run: huggingface-cli login   (paste your token)
    4. Create a new MODEL repo at https://huggingface.co/new
       - Owner: your username (e.g. ubadaghawte)
       - Name: goat-detection-yolov11
       - Type: Model
       - License: MIT
       - Visibility: Public
    5. Update HF_USERNAME below if needed
    6. Run this script
"""

from pathlib import Path
from huggingface_hub import HfApi, upload_file

# ── CONFIG ──────────────────────────────────────────────────────────────────
HF_USERNAME = "ubada11"  # your HuggingFace username
HF_REPO_ID = f"{HF_USERNAME}/goat-detection-yolov11"
MODELS_DIR = Path(__file__).parent.parent / "models"
# ────────────────────────────────────────────────────────────────────────────


def main():
    api = HfApi()

    files_to_upload = [
        "goat_yolo11n_img1024_bs16_lr0.0033_sgd_best.pt",
        "goat_yolo11s_img1024_bs12_lr0.0025_sgd_best.pt",
        "goat_yolo12n_img1024_bs12_lr0.0075_sgd_best.pt",
        "goat_yolo12s_img1024_bs12_lr0.0025_sgd_best.pt",
        "README.md",
    ]

    print(f"\nUploading to: https://huggingface.co/{HF_REPO_ID}\n")

    for filename in files_to_upload:
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            print(f"  SKIP  {filename} (not found at {local_path})")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading  {filename}  ({size_mb:.1f} MB) ...", end=" ", flush=True)

        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print("done")

    print(f"\nAll files uploaded.")
    print(f"View your model at: https://huggingface.co/{HF_REPO_ID}\n")


if __name__ == "__main__":
    main()
