#!/usr/bin/env python3
"""Inspect qk_scale and one-bank Value Embedding behavior from a training checkpoint.

Static inspection needs only a checkpoint. Optional activation inspection also needs
val.bin and the exact model Python file used by the run.
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to ckpt.pt")
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Exact model file used by the run; auto-detected for bundled split/fused versions",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional val.bin path or directory containing val.bin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        cleaned[key] = value
    return cleaned


def tensor_stats(tensor):
    x = tensor.detach().float()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "min": x.min().item(),
        "max": x.max().item(),
        "rms": x.square().mean().sqrt().item(),
    }


def format_values(values):
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def extract_layer_tensors(state_dict, suffix):
    pattern = re.compile(rf"^transformer\.h\.(\d+)\.attn\.{re.escape(suffix)}$")
    found = []
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            found.append((int(match.group(1)), value))
    return sorted(found)


def inspect_static(checkpoint, state_dict):
    result = {
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "final_val_loss": checkpoint.get("final_val_loss"),
        "model_args": checkpoint.get("model_args", {}),
    }
    print("=== Checkpoint ===")
    print(f"iter_num: {result['iter_num']}")
    print(f"best_val_loss: {result['best_val_loss']}")
    print(f"final_val_loss: {result['final_val_loss']}")
    print(f"model_args: {result['model_args']}")

    qk_scales = extract_layer_tensors(state_dict, "qk_scale")
    result["qk_scale"] = {}
    print("\n=== qk_scale by layer/head ===")
    if not qk_scales:
        print("No qk_scale tensors found.")
    for layer, tensor in qk_scales:
        values = tensor.detach().float().tolist()
        stats = tensor_stats(tensor)
        result["qk_scale"][str(layer)] = {"values": values, **stats}
        print(
            f"layer {layer:02d}: {format_values(values)} | "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )

    gates = extract_layer_tensors(state_dict, "value_embed_gate")
    result["value_embed_gate"] = {}
    print("\n=== Value Embedding gate by layer/head ===")
    if not gates:
        print("No value_embed_gate tensors found.")
    for layer, tensor in gates:
        values = tensor.detach().float().tolist()
        stats = tensor_stats(tensor)
        result["value_embed_gate"][str(layer)] = {"values": values, **stats}
        print(
            f"layer {layer:02d}: {format_values(values)} | "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )

    bank = state_dict.get("transformer.vte.weight")
    result["value_embedding_bank"] = None
    if bank is not None:
        n_head = int(result["model_args"].get("n_head", 12))
        if bank.size(1) % n_head != 0:
            raise ValueError("Value Embedding width is not divisible by n_head")
        bank_heads = bank.detach().float().view(bank.size(0), n_head, -1)
        head_rms = bank_heads.square().mean(dim=(0, 2)).sqrt()
        stats = tensor_stats(bank)
        result["value_embedding_bank"] = {
            "shape": list(bank.shape),
            "head_rms": head_rms.tolist(),
            **stats,
        }
        print("\n=== Value Embedding bank ===")
        print(
            f"shape={tuple(bank.shape)} mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f} rms={stats['rms']:.6f} "
            f"min={stats['min']:.6f} max={stats['max']:.6f}"
        )
        print(f"per-head RMS: {format_values(head_rms.tolist())}")

    qkv_layout = "fused" if any(key.endswith(".attn.wqkv.weight") for key in state_dict) else "separate"
    result["qkv_layout"] = qkv_layout
    print(f"\nQKV checkpoint layout: {qkv_layout}")

    optimizer_states = checkpoint.get("optimizers", {})
    adamw_state = optimizer_states.get("adamw", checkpoint.get("optimizer"))
    result["value_embedding_adam"] = None
    if adamw_state:
        for group in adamw_state.get("param_groups", []):
            if group.get("group_name") != "value_embed":
                continue
            params = group.get("params", [])
            if len(params) != 1:
                print(f"Warning: value_embed optimizer group has {len(params)} parameters")
                break
            state = adamw_state.get("state", {}).get(params[0], {})
            adam_result = {"step": str(state.get("step"))}
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    adam_result[name] = tensor_stats(state[name])
            result["value_embedding_adam"] = adam_result
            print("\n=== Value Embedding AdamW state ===")
            print(f"step={adam_result['step']}")
            if "exp_avg" in adam_result:
                print(f"exp_avg RMS={adam_result['exp_avg']['rms']:.8e}")
            if "exp_avg_sq" in adam_result:
                print(f"exp_avg_sq mean={adam_result['exp_avg_sq']['mean']:.8e}")
            break

    return result


def load_model_module(path):
    spec = importlib.util.spec_from_file_location("checkpoint_model_definition", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_val_bin(path):
    return path / "val.bin" if path.is_dir() else path


def raw_value_projection(attn, x_norm):
    if hasattr(attn, "wv"):
        return attn.wv(x_norm)
    if hasattr(attn, "wqkv"):
        return attn.wqkv(x_norm).chunk(3, dim=-1)[2]
    raise AttributeError("Attention module has neither wv nor wqkv")


def resolve_model_file(requested, state_dict):
    if requested is not None:
        return requested.resolve()
    sibling = Path(__file__).resolve().parent
    filename = (
        "model_fusedqkv_value1bank.py"
        if any(key.endswith(".attn.wqkv.weight") for key in state_dict)
        else "model.py"
    )
    return sibling / filename


def inspect_activations(args, checkpoint, state_dict):
    model_file = resolve_model_file(args.model_file, state_dict)
    print(f"Using model definition: {model_file}")
    module = load_model_module(model_file)
    model_args = dict(checkpoint["model_args"])
    valid_fields = set(module.GPTConfig.__dataclass_fields__)
    config = module.GPTConfig(**{key: value for key, value in model_args.items() if key in valid_fields})
    model = module.GPT(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device).eval()

    val_path = resolve_val_bin(args.data.resolve())
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    seq_len = min(args.sequence_length, config.block_size)
    if len(data) <= seq_len + 1:
        raise ValueError("val.bin is shorter than the requested sequence length")
    rng = np.random.default_rng(args.seed)
    layer_records = {}

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        dtype_name = checkpoint.get("config", {}).get("dtype", "bfloat16")
        if dtype_name == "float32":
            amp_context = nullcontext
        else:
            amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
            amp_context = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_context = nullcontext

    with torch.inference_mode():
        for _ in range(args.num_batches):
            starts = rng.integers(0, len(data) - seq_len - 1, size=args.batch_size)
            batch = np.stack([np.asarray(data[i:i + seq_len], dtype=np.int64) for i in starts])
            idx = torch.from_numpy(batch).to(args.device)
            with amp_context():
                pos = torch.arange(seq_len, device=args.device)
                x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
                # The reverted model has no Value Embedding fields; treat them as disabled.
                ve_enabled = bool(getattr(config, 'value_embeds', False))
                ve_start = getattr(config, 'value_embed_start_layer', 0)
                ve_end = ve_start + getattr(config, 'value_embed_num_layers', 0)
                shared_value = model.transformer.vte(idx) if ve_enabled else None
                for layer_idx, block in enumerate(model.transformer.h):
                    selected = ve_enabled and ve_start <= layer_idx < ve_end
                    layer_value = shared_value if selected else None
                    if selected:
                        x_norm = block.ln_1(x)
                        raw_v = raw_value_projection(block.attn, x_norm)
                        B, T, C = raw_v.shape
                        H = config.n_head
                        raw_v = raw_v.view(B, T, H, C // H)
                        # Match the actual attention path: cast the lookup and gate
                        # to V dtype before multiplication, then accumulate metrics
                        # in FP32 for numerical stability.
                        ve = layer_value.to(dtype=raw_v.dtype).view(B, T, H, C // H)
                        gate = block.attn.value_embed_gate.to(dtype=raw_v.dtype).view(1, 1, H, 1)
                        injected = gate * ve
                        raw_sumsq = raw_v.float().square().sum(dim=(0, 1, 3)).cpu()
                        injected_sumsq = injected.float().square().sum(dim=(0, 1, 3)).cpu()
                        count = B * T * (C // H)
                        record = layer_records.setdefault(
                            layer_idx,
                            {
                                "raw_sumsq": torch.zeros(H),
                                "injected_sumsq": torch.zeros(H),
                                "count": 0,
                            },
                        )
                        record["raw_sumsq"] += raw_sumsq
                        record["injected_sumsq"] += injected_sumsq
                        record["count"] += count
                    # Older/split model definitions do not accept a value_embed argument.
                    if selected:
                        x = block(x, value_embed=layer_value)
                    else:
                        x = block(x)

    result = {}
    print("\n=== Activation RMS on validation samples ===")
    print(f"val.bin={val_path} batches={args.num_batches} batch_size={args.batch_size} seq_len={seq_len}")
    for layer, record in sorted(layer_records.items()):
        raw = (record["raw_sumsq"] / record["count"]).sqrt()
        injected = (record["injected_sumsq"] / record["count"]).sqrt()
        ratio = injected / raw.clamp_min(1e-12)
        result[str(layer)] = {
            "raw_value_rms": raw.tolist(),
            "injected_value_rms": injected.tolist(),
            "injected_to_raw_ratio": ratio.tolist(),
        }
        print(f"layer {layer:02d} raw V RMS:       {format_values(raw.tolist())}")
        print(f"layer {layer:02d} injected RMS:    {format_values(injected.tolist())}")
        print(f"layer {layer:02d} injected/raw:    {format_values(ratio.tolist())}")
        print(
            f"layer {layer:02d} ratio summary: mean={ratio.mean().item():.6f} "
            f"min={ratio.min().item():.6f} max={ratio.max().item():.6f}"
        )
    return result


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint.resolve())
    state_dict = clean_state_dict(checkpoint["model"])
    result = inspect_static(checkpoint, state_dict)
    if args.data is not None:
        result["activation_analysis"] = inspect_activations(args, checkpoint, state_dict)
    else:
        print("\nActivation analysis skipped. Pass --data /path/to/openwebtext/val.bin to enable it.")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nWrote JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
