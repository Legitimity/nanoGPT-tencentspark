"""Exp 7: isolate a lower AdamW LR while keeping Muon LR at 0.03."""
from pathlib import Path
import runpy
import sys

PRESET_ARGS = [
    "--lr_schedule=cosine",
    "--learning_rate=0.0024",
    "--muon_lr=0.03",
    "--wandb_run_name=hp-exp7-cosine-adamw2p4e3",
    "--out_dir=out-hp-exp7",
]
provided = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
sys.argv[1:1] = [arg for arg in PRESET_ARGS if arg.split("=", 1)[0] not in provided]
runpy.run_path(str(Path(__file__).with_name("train_exp1.py")), run_name="__main__")
