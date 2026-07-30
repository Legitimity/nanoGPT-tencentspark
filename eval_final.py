"""
Evaluate a trained checkpoint on the *entire* validation set.

estimate_loss() in train.py averages over eval_iters batches of randomly placed
windows, which overlap each other and leave part of the split unvisited. This
script instead walks the split once with non-overlapping windows of block_size
tokens, so every token is predicted exactly once and the result is exact rather
than sampled. The quantity being measured is identical to the training
objective: mean cross entropy over all positions of the window, including the
early ones that have little context. The ragged tail that does not fill a whole
window is dropped.

$ python eval_final.py
$ python eval_final.py --out_dir=out --batch_size=32
"""

import os
import math
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
ckpt_name = 'ckpt.pt'
dataset = 'openwebtext'
split = 'val' # which <split>.bin under data/<dataset> to sweep
batch_size = 16 # windows per forward pass, affects speed but not the result
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16'
compile = False # only worth it when the split is large enough to amortize the ~1min
log_interval = 50 # print progress every N batches
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, ckpt_name)
print(f"loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']
best_val_loss = best_val_loss.item() if torch.is_tensor(best_val_loss) else best_val_loss
print(f"checkpoint is from iteration {iter_num}, sampled best val loss {best_val_loss:.4f}")
checkpoint = None # free up memory

# evaluate at the context length the model was trained with
block_size = model_args['block_size']

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# data
data_path = os.path.join('data', dataset, f'{split}.bin')
data = np.memmap(data_path, dtype=np.uint16, mode='r')
# the window starting at i spans data[i:i+block_size+1], since targets are inputs shifted by one
num_windows = (len(data) - 1) // block_size
assert num_windows > 0, f"{data_path} holds {len(data)} tokens, too few for a single window of {block_size}"
num_tokens = num_windows * block_size
print(f"{data_path}: {len(data):,} tokens -> {num_windows:,} windows of {block_size}, "
      f"scoring {num_tokens:,} tokens ({100*num_tokens/len(data):.2f}%), "
      f"dropping {len(data)-num_tokens:,} tail tokens")

@torch.no_grad()
def evaluate():
    loss_sum = 0.0 # cross entropy summed over every scored token
    t0 = time.time()
    for batch_idx, start in enumerate(range(0, num_windows, batch_size)):
        offsets = [i * block_size for i in range(start, min(start + batch_size, num_windows))]
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in offsets])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in offsets])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        # loss is a mean over this batch, so weight it by the tokens it covers to
        # keep a possibly smaller final batch from being over-counted
        loss_sum += loss.item() * y.numel()
        if log_interval > 0 and batch_idx % log_interval == 0:
            done = min(start + batch_size, num_windows)
            dt = time.time() - t0
            eta = dt / done * (num_windows - done)
            print(f"  {done:,}/{num_windows:,} windows, running loss {loss_sum/(done*block_size):.4f}, eta {eta:.0f}s")
    return loss_sum / num_tokens, time.time() - t0

loss, elapsed = evaluate()
print(f"\n{split} loss {loss:.4f} | perplexity {math.exp(loss):.4f} | bits per token {loss/math.log(2):.4f}")
print(f"over all {num_tokens:,} tokens in {elapsed:.1f}s ({num_tokens/elapsed:,.0f} tok/s)")
