"""
Full-pass validation loss evaluation for a checkpoint.

Computes the exact mean loss over non-overlapping windows of val.bin
(every token predicted exactly once), instead of a Monte Carlo estimate.

Optionally attaches the result to an existing wandb run (updates both the
run summary and the run history at the checkpoint's iter_num).

Usage:
  python3 eval_full.py --ckpt=out/ckpt.pt
  python3 eval_full.py --ckpt=out/ckpt.pt \
      --wandb_entity=jqh333 --wandb_project=owt --wandb_run_id=lj4lkgge
"""

import os
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='out/ckpt.pt', help='path to checkpoint')
parser.add_argument('--data_dir', default='data/openwebtext', help='directory containing val.bin')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--device', default='cuda')
parser.add_argument('--metric', default='val/final_loss_full', help='wandb metric name')
parser.add_argument('--wandb_entity', default=None)
parser.add_argument('--wandb_project', default=None)
parser.add_argument('--wandb_run_id', default=None, help='existing run id to attach the metric to')
args = parser.parse_args()

device_type = 'cuda' if 'cuda' in args.device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load model from checkpoint
checkpoint = torch.load(args.ckpt, map_location=args.device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()

# full sequential pass over val.bin with non-overlapping windows
block_size = model_args['block_size']
data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
n_windows = (len(data) - 1) // block_size
print(f"evaluating {args.ckpt} (iter {checkpoint['iter_num']}): {n_windows} windows of {block_size} tokens")

total_loss, total_windows = 0.0, 0
with torch.no_grad():
    for i in range(0, n_windows, args.batch_size):
        hi = min(i + args.batch_size, n_windows)
        x = torch.stack([torch.from_numpy(data[j*block_size:(j+1)*block_size].astype(np.int64)) for j in range(i, hi)])
        y = torch.stack([torch.from_numpy(data[j*block_size+1:(j+1)*block_size+1].astype(np.int64)) for j in range(i, hi)])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
        else:
            x, y = x.to(args.device), y.to(args.device)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item() * (hi - i)  # equal-length windows, weight by window count
        total_windows += hi - i
val_loss = total_loss / total_windows
print(f"full val loss: {val_loss:.6f}")

# attach result to an existing wandb run
if args.wandb_run_id:
    import wandb
    run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}"
    api = wandb.Api()
    run = api.run(run_path)
    run.summary[args.metric] = val_loss
    run.summary.update()
    print(f"summary updated: {run_path} -> {args.metric} = {val_loss:.6f}")
    # also append to history at the checkpoint's step (reopens the finished run briefly)
    try:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                   id=args.wandb_run_id, resume='must')
        wandb.log({args.metric: val_loss}, step=checkpoint['iter_num'])
        wandb.finish()
        print(f"history appended at step {checkpoint['iter_num']}")
    except Exception as e:
        print(f"history append skipped ({type(e).__name__}: {e}); summary already updated")
