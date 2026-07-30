"""Exp 11: halve Muon weight decay at the current best cosine peak LRs."""
from pathlib import Path
import runpy
import sys

PRESET_ARGS = [
    "--lr_schedule=cosine",
    "--learning_rate=0.003",
    "--muon_lr=0.03",
    "--muon_weight_decay=0.005",
    "--wandb_run_name=hp-exp11-cosine-muon-wd5e3",
    "--out_dir=out-hp-exp11",
]
provided = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
sys.argv[1:1] = [arg for arg in PRESET_ARGS if arg.split("=", 1)[0] not in provided]
runpy.run_path(str(Path(__file__).with_name("train_exp1.py")), run_name="__main__")
