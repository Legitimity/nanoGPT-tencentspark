"""Exp 12: lower both peak LRs with WSD linear cooldown over the final 20%."""
from pathlib import Path
import runpy
import sys

PRESET_ARGS = [
    "--lr_schedule=wsd",
    "--wsd_decay_frac=0.2",
    "--wsd_decay_style=linear",
    "--learning_rate=0.0024",
    "--muon_lr=0.024",
    "--wandb_run_name=hp-exp12-wsd-linear-f20-lower-peaks",
    "--out_dir=out-hp-exp12",
]
provided = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
sys.argv[1:1] = [arg for arg in PRESET_ARGS if arg.split("=", 1)[0] not in provided]
runpy.run_path(str(Path(__file__).with_name("train_exp1.py")), run_name="__main__")
