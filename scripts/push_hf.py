"""Upload an exported model folder to the Hugging Face Hub, with a rendered model card.

    uv run hf auth login                       # once, with a write token
    uv run python scripts/push_hf.py --folder out/dassie-50m --repo khkstudios/dassie-50m
    uv run python scripts/push_hf.py --dry-run # render README.md into the folder, upload nothing
"""
import argparse
import shutil
from pathlib import Path

from render_card import render

p = argparse.ArgumentParser()
p.add_argument("--folder", default="out/dassie-50m")
p.add_argument("--repo", default="khkstudios/dassie-50m")
p.add_argument("--card", default="model_card.md")
p.add_argument("--ckpt", default="checkpoints/50m/best.pt")
p.add_argument("--no-samples", action="store_true", help="skip generating sample stories for the card")
p.add_argument("--private", action="store_true")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

folder = Path(args.folder)
if not (folder / "model.safetensors").exists():
    if not args.dry_run:
        raise SystemExit(f"no exported model in {folder}/ - run scripts/export_hf.py first "
                         f"(or use --dry-run to preview the card)")
    print(f"note: no exported model in {folder}/ yet, so metrics and samples will show as _TBD_")
folder.mkdir(parents=True, exist_ok=True)
card = render(Path(args.card).read_text(), folder, args.ckpt, args.repo, not args.no_samples)
(folder / "README.md").write_text(card)
(folder / "assets").mkdir(exist_ok=True)
shutil.copy("assets/dassie.svg", folder / "assets" / "dassie.svg")
log = Path(args.ckpt).parent / "log.csv"
if log.exists():
    shutil.copy(log, folder / "training_log.csv")

if args.dry_run:
    print(f"dry run: card rendered to {folder / 'README.md'}, nothing uploaded")
    raise SystemExit

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model",
                  commit_message="Upload dassie-50m")
print(f"https://huggingface.co/{args.repo}")
