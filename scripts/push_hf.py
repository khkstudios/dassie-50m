"""Upload an exported model folder to the Hugging Face Hub.

    uv run hf auth login                       # once, with a write token
    uv run python scripts/push_hf.py --folder out/dassie-50m --repo keegalix/dassie-50m
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--folder", default="out/dassie-50m")
p.add_argument("--repo", default="keegalix/dassie-50m")
p.add_argument("--card", default="model_card.md")
p.add_argument("--private", action="store_true")
args = p.parse_args()

folder = Path(args.folder)
shutil.copy(args.card, folder / "README.md")
for extra in ["checkpoints/50m/log.csv"]:
    if Path(extra).exists():
        shutil.copy(extra, folder / "training_log.csv")

api = HfApi()
api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                  commit_message="Upload dassie-50m")
print(f"https://huggingface.co/{args.repo}")
