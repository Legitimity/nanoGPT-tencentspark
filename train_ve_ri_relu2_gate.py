"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_ve_ri_relu2_gate import GPTConfig, GPT, Muon

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
always_save_checkpoint = True # if True, save a checkpoint every save_interval iterations
save_interval = 500 # save a checkpoint every N iterations (only when always_save_checkpoint is True)
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

#eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10
eval_only = False # if True, script exits right after the first eval

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
qk_norm = True # whether to RMS-normalize q,k per head before attention (QK-Norm)
qk_scale = True # whether to learn a per-head temperature multiplied onto q (after QK-Norm)
fused_qkv = True # single forward GEMM; Muon keeps Q/K/V logical matrices separate
ce_chunk_size = 0 # chunked CE disabled (slower under compile); model file accepts the arg
# one-bank Value Embeddings: one token table shared by the final four layers
value_embeds = True
value_embed_start_layer = 8 # zero-based
value_embed_num_layers = 4 # exactly layers 8,9,10,11 receive the shared bank
value_embed_gate_init = 1.0 # additive per-layer/per-head gain
value_embed_init_std = 0.02 # small relative to the initial projected V RMS
value_embed_lr_scale = 1.0 # relative to AdamW learning_rate; no weight decay
resid_init_scale = 0.1 # residual-projection init std multiplier (much smaller than default 1.0)
# adamw optimizer
learning_rate = 1e-3 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 8e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
lr_schedule = 'cosine' # 'cosine' or 'wsd' (warmup-stable-decay / trapezoidal)
wsd_decay_frac = 0.2 # WSD: fraction of lr_decay_iters used for the final decay (annealing) phase
wsd_decay_style = 'linear' # WSD: decay shape, 'linear' or 'cosine'
# optimizer selection
optimizer_name = 'muon' # 'adamw' or 'muon' (Muon for 2D attn/FFN matrices, AdamW for embeddings & 1D params)
muon_lr = 0.02 # Muon learning rate (larger than adamw lr since Muon updates are orthogonalized)
muon_momentum = 0.95 # Muon momentum
muon_weight_decay = 0.005 # weight decay for Muon params; calibrated so lr*wd shrink rate matches the AdamW side (0.02*0.005 = 1e-3*0.1)
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
# Scale eval_iters inversely with micro-batch size so each evaluation covers the same
# number of tokens as the batch_size=6 baseline (eval_iters=20 -> ~123K tokens).
# This keeps eval wall-time and val/loss noise constant across batch-size experiments.
eval_iters = max(1, round(eval_iters * 6 / batch_size))
print(f"eval: {eval_iters} iters x batch_size {batch_size} x {block_size} tokens per split")
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank = 0
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, qk_norm=qk_norm,
    qk_scale=qk_scale, fused_qkv=fused_qkv, ce_chunk_size=ce_chunk_size,
    value_embeds=value_embeds, value_embed_start_layer=value_embed_start_layer,
    value_embed_num_layers=value_embed_num_layers,
    value_embed_gate_init=value_embed_gate_init,
    value_embed_init_std=value_embed_init_std,
    resid_init_scale=resid_init_scale,
) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Force architecture and forward-semantics attributes to match the
    # checkpoint. qk_norm has no state_dict tensor, so load_state_dict cannot
    # detect a mismatch for us; it must be restored explicitly.
    checkpoint_model_keys = [
        'n_layer', 'n_head', 'n_embd', 'block_size',
        'bias', 'vocab_size', 'qk_norm', 'qk_scale', 'fused_qkv',
        'value_embeds', 'value_embed_start_layer', 'value_embed_num_layers',
        'value_embed_gate_init', 'value_embed_init_std',
    ]
    missing_model_keys = [
        k for k in checkpoint_model_keys if k not in checkpoint_model_args
    ]
    if missing_model_keys:
        raise KeyError(
            "checkpoint model_args is missing required keys: "
            f"{missing_model_keys}. Automatic legacy fused-QKV checkpoint "
            "migration is intentionally not performed."
        )
    for k in checkpoint_model_keys:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout, qk_norm=qk_norm, qk_scale=qk_scale)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in [
        'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
        'qk_norm', 'qk_scale', 'fused_qkv',
    ]:
        model_args[k] = getattr(model.config, k)
# Cropping a resumed model would leave the saved AdamW state for position
# embeddings at the old shape. Require an exact block size for safe resume.
if init_from == 'resume' and block_size != model.config.block_size:
    raise ValueError(
        f"resume requires block_size={model.config.block_size}, got {block_size}"
    )
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
if optimizer_name not in {'muon', 'adamw'}:
    raise ValueError(f"unsupported optimizer_name: {optimizer_name!r}")
if optimizer_name == 'muon':
    # Muon is restricted to 2D hidden-layer matrices. Everything else stays
    # on AdamW, including tied token/output embeddings, position embeddings,
    # LayerNorm gains, and biases.
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    muon_named = [
        (n, p) for n, p in named_params
        if p.ndim == 2 and n.startswith('transformer.h.')
    ]
    qkv_muon_named = [(n, p) for n, p in muon_named if n.endswith('.attn.wqkv.weight')]
    regular_muon_named = [(n, p) for n, p in muon_named if not n.endswith('.attn.wqkv.weight')]
    muon_ids = {id(p) for _, p in muon_named}
    qkv_muon_params = [p for _, p in qkv_muon_named]
    regular_muon_params = [p for _, p in regular_muon_named]
    assert len(qkv_muon_params) == model.config.n_layer, "expected exactly one fused QKV matrix per layer"
    assert all(
        p.shape == (3 * model.config.n_embd, model.config.n_embd)
        for p in qkv_muon_params
    )
    adam_decay = [
        p for _, p in named_params if id(p) not in muon_ids and p.ndim >= 2
    ]
    adam_nodecay = [
        p for _, p in named_params if id(p) not in muon_ids and p.ndim < 2
    ]

    # Fail early if a future model edit duplicates or omits a parameter.
    all_ids = {id(p) for _, p in named_params}
    adam_ids = {id(p) for p in adam_decay + adam_nodecay}
    assert muon_ids.isdisjoint(adam_ids), "Muon and AdamW parameter groups overlap"
    assert muon_ids | adam_ids == all_ids, "Optimizer parameter partition is incomplete"
    assert all(p.ndim == 2 for _, p in muon_named), "Muon received a non-2D parameter"

    logical_muon_matrices = len(regular_muon_params) + 3 * len(qkv_muon_params)
    muon_parameter_count = sum(p.numel() for _, p in muon_named)
    print(
        f"muon: {logical_muon_matrices} logical matrices in "
        f"{len(regular_muon_params) + len(qkv_muon_params)} tensors, "
        f"{muon_parameter_count/1e6:.2f}M params | "
        f"adamw: {sum(p.numel() for p in adam_decay + adam_nodecay)/1e6:.2f}M params"
    )
    adam_groups = [
        {'params': adam_decay, 'weight_decay': weight_decay},
        {'params': adam_nodecay, 'weight_decay': 0.0},
    ]
    muon_groups = [
        {'params': regular_muon_params, 'muon_split_rows': 1},
        {'params': qkv_muon_params, 'muon_split_rows': 3},
    ]
    optimizers = [
        Muon(muon_groups, lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay),
        torch.optim.AdamW(
            adam_groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8,
            fused=(device_type == 'cuda'),
        ),
    ]
else:
    optimizers = [model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type,
    )]
# record base_lr so the lr schedule can scale all param groups proportionally
for opt in optimizers:
    for param_group in opt.param_groups:
        param_group.setdefault('base_lr', param_group['lr'])
if init_from == 'resume':
    saved_optimizer_name = checkpoint.get(
        'optimizer_name',
        checkpoint.get('config', {}).get('optimizer_name'),
    )
    if saved_optimizer_name is not None and saved_optimizer_name != optimizer_name:
        raise ValueError(
            f"checkpoint optimizer is {saved_optimizer_name!r}, "
            f"but current optimizer_name is {optimizer_name!r}"
        )
    optimizer_states = checkpoint.get('optimizers', {})
    adamw_state = optimizer_states.get('adamw', checkpoint.get('optimizer'))
    if adamw_state is None:
        raise KeyError("checkpoint does not contain AdamW optimizer state")
    optimizers[-1].load_state_dict(adamw_state)
    if optimizer_name == 'muon':
        muon_state = optimizer_states.get('muon', checkpoint.get('optimizer_muon'))
        if muon_state is None:
            raise KeyError("checkpoint does not contain Muon optimizer state")
        optimizers[0].load_state_dict(muon_state)
    # Older checkpoints may predate base_lr. Loading an optimizer state replaces
    # its param-group dictionaries, so restore only the missing metadata here.
    for opt_index, opt in enumerate(optimizers):
        fallback_base_lr = (
            muon_lr if optimizer_name == 'muon' and opt_index == 0
            else learning_rate
        )
        for param_group in opt.param_groups:
            param_group.setdefault('base_lr', fallback_base_lr)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    # Evaluation runs only on the master process. Bypass the DDP wrapper so
    # these forwards cannot trigger collectives that the other ranks do not join.
    raw_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = raw_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    raw_model.train()
    return out

# full-coverage validation loss: one sequential pass over non-overlapping windows of val.bin
@torch.no_grad()
def estimate_val_full():
    raw_model.eval()
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    n_windows = (len(data) - 1) // block_size
    total_loss, total_windows = 0.0, 0
    for i in range(0, n_windows, batch_size):
        hi = min(i + batch_size, n_windows)
        x = torch.stack([torch.from_numpy(data[j*block_size:(j+1)*block_size].astype(np.int64)) for j in range(i, hi)])
        y = torch.stack([torch.from_numpy(data[j*block_size+1:(j+1)*block_size+1].astype(np.int64)) for j in range(i, hi)])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = raw_model(x, y)
        total_loss += loss.item() * (hi - i) # each window has equal length, so weight by window count
        total_windows += hi - i
    raw_model.train()
    return total_loss / total_windows

# learning rate decay scheduler (cosine or wsd/trapezoidal, with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    if lr_schedule == 'wsd':
        # 3a) warmup-stable-decay: constant lr until the last wsd_decay_frac of steps, then decay to min_lr
        decay_start = lr_decay_iters * (1.0 - wsd_decay_frac)
        if it < decay_start:
            return learning_rate # stable phase
        decay_ratio = (it - decay_start) / (lr_decay_iters - decay_start)
        assert 0 <= decay_ratio <= 1
        coeff = (1.0 - decay_ratio) if wsd_decay_style == 'linear' else 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    # 3b) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(entity='jqh333', project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

def save_training_checkpoint(filename='ckpt.pt', final_val_loss=None):
    """Atomically save model plus every optimizer state on the master rank."""
    if not master_process:
        return
    optimizer_states = {
        'adamw': optimizers[-1].state_dict(),
        'muon': optimizers[0].state_dict() if optimizer_name == 'muon' else None,
    }
    checkpoint = {
        'model': raw_model.state_dict(),
        # Keep the legacy keys so older resume scripts can still read this file.
        'optimizer': optimizer_states['adamw'],
        'optimizer_muon': optimizer_states['muon'],
        'optimizers': optimizer_states,
        'optimizer_name': optimizer_name,
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    if final_val_loss is not None:
        checkpoint['final_val_loss'] = final_val_loss
    checkpoint_path = os.path.join(out_dir, filename)
    temporary_path = checkpoint_path + '.tmp'
    print(f"saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

# Preserve nanoGPT's historical inclusive interpretation: max_iters is the
# largest update index, so a fresh run executes updates 0, ..., max_iters.
# Writing the target explicitly removes the accidental-looking `> max_iters`
# termination test without changing the numerical training trajectory.
target_num_updates = max_iters + 1
did_train = False
while iter_num < target_num_updates:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    lr_scale = lr / learning_rate # schedule scale factor, applied to each group's base_lr
    for opt in optimizers:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['base_lr'] * lr_scale

    # evaluate the loss on train/val sets
    if (eval_only or iter_num % eval_interval == 0) and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
        if wandb_log:
            raw_model.eval()
            X_stats, Y_stats = get_batch('val')
            with torch.no_grad(), ctx:
                _, _, head_stats = raw_model(X_stats, Y_stats, collect_head_stats=True)
            raw_model.train()
            wandb_metrics = {
                "val/loss": losses['val'],
                "lr": lr,
                "lr/adamw": optimizers[-1].param_groups[0]['lr'],
                "mfu": running_mfu*100, # convert to percentage
            }
            if optimizer_name == 'muon':
                wandb_metrics["lr/muon"] = optimizers[0].param_groups[0]['lr']
            layer_stats = head_stats.mean(dim=1)
            for layer_idx in range(layer_stats.size(0)):
                prefix = f"Charts/hidden_states/layer_{layer_idx}"
                wandb_metrics[f"{prefix}/rms.avg"] = layer_stats[layer_idx, 0].item()
                wandb_metrics[f"{prefix}/rms.max"] = layer_stats[layer_idx, 1].item()
                wandb_metrics[f"{prefix}/abs.max"] = layer_stats[layer_idx, 2].item()
            wandb.log(wandb_metrics, step=iter_num)

    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    loss_accum = torch.zeros((), device=device) if master_process else None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        if master_process:
            loss_accum += loss.detach()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # collect the pre-clipping gradient norm, then clip if enabled
    # muon/adamw params form a disjoint partition, so unscaling each optimizer scales every grad exactly once
    for opt in optimizers:
        scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip if grad_clip > 0.0 else float('inf')
    )
    # step the optimizer(s) and scaler if training in fp16
    for opt in optimizers:
        scaler.step(opt)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # Convert the average accumulated loss to a CPU value for logging.
    lossf = loss_accum.item() if master_process else None
    if iter_num % log_interval == 0 and master_process:
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    if wandb_log and master_process:
        wandb_metrics = {
            "train/lm_loss": lossf,
            "train/grad_norm": grad_norm.item(),
            "lr": lr,
            "lr/adamw": optimizers[-1].param_groups[0]['lr'],
        }
        if optimizer_name == 'muon':
            wandb_metrics["lr/muon"] = optimizers[0].param_groups[0]['lr']
        wandb.log(wandb_metrics, step=iter_num)
    iter_num += 1
    local_iter_num += 1
    did_train = True

    # Saving after the update makes iter_num mean "number of completed
    # updates". At multiples of save_interval this is numerically the same
    # model state that the old code saved before the next indexed update.
    if always_save_checkpoint and iter_num % save_interval == 0:
        save_training_checkpoint()

# final evaluation: full sequential pass over the entire validation set
if master_process:
    final_val_loss = estimate_val_full()
    print(f"final val loss (full pass over val.bin): {final_val_loss:.4f}")
    if wandb_log:
        wandb.log({"val/final_loss_full": final_val_loss}, step=iter_num)
        wandb.run.summary["val/final_loss_full"] = final_val_loss
    if did_train:
        # Persist the exact model state whose final validation loss is reported.
        save_training_checkpoint(final_val_loss=final_val_loss)

if ddp:
    destroy_process_group()