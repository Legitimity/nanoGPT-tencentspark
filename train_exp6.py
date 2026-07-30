"""Exp 6: WSD cosine cooldown over the final 20% at the current best peak LRs."""
from pathlib import Path
import runpy
import sys

PRESET_ARGS = [
    "--lr_schedule=wsd",
    "--wsd_decay_frac=0.2",
    "--wsd_decay_style=cosine",
    "--learning_rate=0.003",
    "--muon_lr=0.03",
    "--wandb_run_name=hp-exp6-wsd-cosine-f20-a3e3-m3e2",
    "--out_dir=out-hp-exp6",
]
provided = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
sys.argv[1:1] = [arg for arg in PRESET_ARGS if arg.split("=", 1)[0] not in provided]
runpy.run_path(str(Path(__file__).with_name("train_exp1.py")), run_name="__main__")
