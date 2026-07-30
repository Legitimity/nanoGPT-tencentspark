"""Exp 10: isolate a higher Muon LR while keeping AdamW LR at 0.003."""
from pathlib import Path
import runpy
import sys

PRESET_ARGS = [
    "--lr_schedule=cosine",
    "--learning_rate=0.003",
    "--muon_lr=0.035",
    "--wandb_run_name=hp-exp10-cosine-muon3p5e2",
    "--out_dir=out-hp-exp10",
]
provided = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
sys.argv[1:1] = [arg for arg in PRESET_ARGS if arg.split("=", 1)[0] not in provided]
runpy.run_path(str(Path(__file__).with_name("train_exp1.py")), run_name="__main__")
