"""
Full definition of a GPT Language Model, all of it in this single file.
MLP-output-zero-init ablation of the current best dense-FFN speedrun model.
It preserves Value Embeddings, LayerNorm, gated attention, ReLU² FFN and
readout backout. Only each MLP c_proj starts at zero; attention c_proj keeps
the existing tiny residual initialization.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    # P0: fused linear cross-entropy computes CE(lm_head(x), y) without
    # materializing the [B*T, V] logits (largest single speed item in list.md).
    # CCE handles the fp32-weight/bf16-hidden mix internally (liger-kernel
    # 0.8.1 crashed with illegal memory access in repeated compiled backward).
    from cut_cross_entropy import linear_cross_entropy as _CCE
except Exception:
    _CCE = None

if _CCE is not None:
    # Keep the fused CE in a dynamo-disabled island: third-party custom ops are
    # opaque to dynamo, and this guarantees stable repeated backward while the
    # model body stays compiled.
    @torch._dynamo.disable
    def _cce_loss(weight, hidden, targets, dt):
        return _CCE(
            hidden.to(dt).contiguous(),
            weight,
            targets.contiguous(),
            ignore_index=-1,
            reduction='mean',
        )

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # One GEMM produces Q, K, V and the attention output gate. The optimizer
        # treats the four row blocks as independent square matrices during Muon
        # orthogonalization (muon_split_rows = 4).
        self.wqkv = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.qk_norm = config.qk_norm
        # learnable per-head temperature multiplied onto q (after QK-Norm, which would
        # otherwise cancel any scalar); lets each head tune its logit sharpness
        self.qk_scale = nn.Parameter(torch.ones(config.n_head)) if config.qk_scale else None
        # A single token-indexed Value Embedding bank is shared by the selected
        # upper layers. Each selected layer learns only a per-head gain.
        value_embed_end_layer = config.value_embed_start_layer + config.value_embed_num_layers
        self.use_value_embed = (
            config.value_embeds
            and config.value_embed_start_layer <= layer_idx < value_embed_end_layer
        )
        self.value_embed_gate = (
            nn.Parameter(torch.full((config.n_head,), config.value_embed_gate_init))
            if self.use_value_embed else None
        )
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, value_embed=None, collect_head_stats=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # A single projection/GEMM produces Q, K, V and the gate in contiguous
        # row blocks. The gate sees the same input, exactly like exp4's separate
        # bias-free gate Linear.
        q, k, v, g = self.wqkv(x).split(C, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.use_value_embed:
            assert value_embed is not None, "selected Value Embedding layer received no shared bank lookup"
            assert value_embed.shape == (B, T, C)
            # Embedding lookup remains FP32 under autocast. Cast before mixing so
            # SDPA stays on its BF16/FP16 fast path instead of being promoted.
            ve = value_embed.to(dtype=v.dtype).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            gate = self.value_embed_gate.to(dtype=v.dtype).view(1, self.n_head, 1, 1)
            v = v + gate * ve
        else:
            assert value_embed is None, "Value Embedding was passed to a layer outside the injection range"
        if self.qk_norm:
            # QK-Norm: parameter-free RMS normalization per head, keeps attention logits
            # at controlled scale and prevents logit blow-up during high-lr phases
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
        if self.qk_scale is not None:
            # learnable per-head temperature; must be applied after rms_norm (a scalar
            # before the norm would be normalized away)
            qk_scale = self.qk_scale.to(dtype=q.dtype).view(1, -1, 1, 1)
            q = q * qk_scale

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        head_stats = None
        if collect_head_stats:
            y_float = y.detach().float()
            token_rms = y_float.square().mean(dim=-1).sqrt()
            head_stats = torch.stack([
                token_rms.mean(dim=(0, 2)),
                token_rms.amax(dim=(0, 2)),
                y_float.abs().amax(dim=(0, 2, 3)),
            ], dim=-1)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = y * torch.sigmoid(g) # gated attention (Qiu et al. 2024 / speedrun)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if collect_head_stats:
            return y, head_stats
        return y

class MLP(nn.Module):
    # Dense FFN with relu**2 activation (matches model_ve_ri_relu2_gate.py).

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # ReLU^2 (square() avoids autocast's fp32 promotion of pow())
        x = self.c_proj(x)
        x = self.dropout(x)
        return x




class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, value_embed=None, collect_head_stats=False):
        if collect_head_stats:
            attn_out, head_stats = self.attn(
                self.ln_1(x), value_embed=value_embed, collect_head_stats=True
            )
        else:
            attn_out = self.attn(self.ln_1(x), value_embed=value_embed)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if collect_head_stats:
            return x, head_stats
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    qk_norm: bool = True # whether to RMS-normalize q and k per head before attention (QK-Norm)
    qk_scale: bool = True # whether to learn a per-head temperature multiplied onto q (after QK-Norm)
    fused_qkv: bool = True # one forward GEMM; Muon still orthogonalizes Q/K/V row blocks independently
    ce_chunk_size: int = 0 # unused in this model file; accepted for train.py model_args compatibility
    value_embeds: bool = False # one shared token-indexed Value Embedding bank
    value_embed_start_layer: int = 8 # zero-based first injection layer
    value_embed_num_layers: int = 4 # exactly layers 8,9,10,11 in the 12-layer experiment
    value_embed_gate_init: float = 1.0 # per-layer, per-head additive gain initialization
    value_embed_init_std: float = 0.02 # bank initialization, matched to token embeddings
    resid_init_scale: float = 1.0 # multiplier on the GPT-2 residual-projection init std (0.02/sqrt(2*n_layer))
    fused_ce: bool = True # training-only: cut-cross-entropy fused linear CE (no logits materialization); eval always uses the plain path
    readout_backout: bool = True # mix an earlier normalized residual stream into final readout
    readout_backout_layer: int = 7 # zero-based block output cached for the skip readout
    readout_backout_final_init: float = 1.0 # baseline final-stream coefficient
    readout_backout_skip_init: float = 0.0 # earlier-stream coefficient starts closed

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.fused_qkv, "this model file implements only the fused-QKV layout"
        if config.readout_backout:
            assert 0 <= config.readout_backout_layer < config.n_layer - 1
        if config.value_embeds:
            assert config.value_embed_num_layers > 0
            assert 0 <= config.value_embed_start_layer < config.n_layer
            assert config.value_embed_start_layer + config.value_embed_num_layers <= config.n_layer
        self.config = config

        transformer_modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.value_embeds:
            transformer_modules['vte'] = nn.Embedding(config.vocab_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.readout_backout_mix = (
            nn.Parameter(torch.tensor([
                config.readout_backout_final_init,
                config.readout_backout_skip_init,
            ]))
            if config.readout_backout else None
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        if config.value_embeds:
            torch.nn.init.normal_(
                self.transformer.vte.weight,
                mean=0.0,
                std=config.value_embed_init_std,
            )
        # Keep attention residual projections at the current tiny-init scale, but
        # start every MLP residual branch exactly closed. This changes only the
        # initialization; parameter shapes and the runtime graph are unchanged.
        for pn, p in self.named_parameters():
            if pn.endswith('.mlp.c_proj.weight'):
                torch.nn.init.zeros_(p)
            elif pn.endswith('.attn.c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer) * config.resid_init_scale,
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, collect_head_stats=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        # One lookup is shared by all selected upper layers. Do not duplicate the
        # vocabulary-sized parameter bank per layer.
        shared_value_embed = self.transformer.vte(idx) if self.config.value_embeds else None
        if shared_value_embed is not None and idx.device.type == 'cuda' and torch.is_autocast_enabled():
            # Cast once for all selected layers. The per-attention cast below is then
            # a no-op and protects non-autocast/custom-dtype callers.
            shared_value_embed = shared_value_embed.to(dtype=torch.get_autocast_gpu_dtype())
        value_embed_end_layer = self.config.value_embed_start_layer + self.config.value_embed_num_layers
        all_head_stats = []
        backout_hidden = None
        for layer_idx, block in enumerate(self.transformer.h):
            layer_value_embed = (
                shared_value_embed
                if self.config.value_embeds
                and self.config.value_embed_start_layer <= layer_idx < value_embed_end_layer
                else None
            )
            if collect_head_stats:
                x, head_stats = block(
                    x, value_embed=layer_value_embed, collect_head_stats=True
                )
                all_head_stats.append(head_stats)
            else:
                x = block(x, value_embed=layer_value_embed)
            if self.config.readout_backout and layer_idx == self.config.readout_backout_layer:
                backout_hidden = x
        if self.readout_backout_mix is not None:
            assert backout_hidden is not None
            final_norm = self.transformer.ln_f(x)
            skip_norm = self.transformer.ln_f(backout_hidden)
            mix = self.readout_backout_mix.to(dtype=final_norm.dtype)
            x = mix[0] * final_norm + mix[1] * skip_norm
        else:
            x = self.transformer.ln_f(x)

        if targets is not None:
            if self.training and self.config.fused_ce and _CCE is not None and x.device.type == 'cuda':
                # P0: fused linear CE — identical math, never materializes logits.
                # Training-only; eval keeps the plain path so reported losses stay
                # numerically comparable with earlier experiment runs.
                dt = torch.get_autocast_gpu_dtype() if (x.device.type == 'cuda' and torch.is_autocast_enabled('cuda')) else x.dtype
                logits = None
                loss = _cce_loss(self.lm_head.weight, x.view(-1, x.size(-1)), targets.view(-1), dt)
            else:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if collect_head_stats:
            return logits, loss, torch.stack(all_head_stats)
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # These options add no incompatible pretrained tensors. A Value Embedding
        # bank and its gates are newly initialized after GPT-2 weights are copied.
        allowed_overrides = {
            'dropout', 'qk_norm', 'qk_scale', 'value_embeds', 'value_embed_start_layer',
            'value_embed_num_layers', 'value_embed_gate_init', 'value_embed_init_std',
            'readout_backout', 'readout_backout_layer',
            'readout_backout_final_init', 'readout_backout_skip_init',
            'resid_init_scale',
        }
        assert all(k in allowed_overrides for k in override_args), \
            f"unsupported override keys: {set(override_args) - allowed_overrides}"
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # HF checkpoints carry no qk_scale parameter. It may be enabled as a
        # newly initialized unit scale without changing imported tensor shapes.
        config_args['qk_scale'] = override_args.get('qk_scale', False)
        # We can override dropout and QK-Norm without affecting weight shapes.
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        if 'qk_norm' in override_args:
            print(f"overriding QK-Norm to {override_args['qk_norm']}")
            config_args['qk_norm'] = override_args['qk_norm']
        for key in [
            'value_embeds', 'value_embed_start_layer', 'value_embed_num_layers',
            'value_embed_gate_init', 'value_embed_init_std',
            'readout_backout', 'readout_backout_layer',
            'readout_backout_final_init', 'readout_backout_skip_init',
            'resid_init_scale',
        ]:
            if key in override_args:
                print(f"overriding {key} to {override_args[key]}")
                config_args[key] = override_args[key]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        copied = set()
        for k in sd_keys_hf:
            if k.endswith('attn.c_attn.weight'):
                # Hugging Face Conv1D stores [in, 3*out]; nn.Linear stores
                # [3*out, in]. Q/K/V row order is already identical. The fused
                # wqkv here also carries a fourth row block for the attention
                # output gate, which stays newly initialized.
                key = k.replace('c_attn', 'wqkv')
                nx, nf = sd_hf[k].shape # Conv1D stores [in=C, out=3C]
                assert nf == 3 * nx and sd[key].shape == (4 * nx, nx)
                with torch.no_grad():
                    sd[key][:nf].copy_(sd_hf[k].t())
                copied.add(key)
            elif k.endswith('attn.c_attn.bias'):
                key = k.replace('c_attn', 'wqkv')
                C3 = sd_hf[k].shape[0]
                assert sd[key].shape == (4 * C3 // 3,)
                with torch.no_grad():
                    sd[key][:C3].copy_(sd_hf[k])
                copied.add(key)
            elif any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                copied.add(k)
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                copied.add(k)
        newly_initialized = {
            k for k in sd_keys
            if k == 'transformer.vte.weight'
            or k.endswith('.attn.value_embed_gate')
            or k.endswith('.attn.qk_scale')
            or k == 'readout_backout_mix'
        }
        expected_copied = set(sd_keys) - newly_initialized
        assert copied == expected_copied, f"uncopied model keys: {expected_copied - copied}"

        return model

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type,
        value_embed_lr_scale=1.0,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # The large Value Embedding bank is lookup-based rather than a matmul
        # matrix. Keep it out of weight decay and expose an independent LR scale.
        value_embed_params = [
            p for n, p in param_dict.items() if n == 'transformer.vte.weight'
        ]
        value_embed_ids = {id(p) for p in value_embed_params}
        decay_params = [
            p for _, p in param_dict.items()
            if id(p) not in value_embed_ids and p.dim() >= 2
        ]
        nodecay_params = [
            p for _, p in param_dict.items()
            if id(p) not in value_embed_ids and p.dim() < 2
        ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        if value_embed_params:
            optim_groups.append({
                'params': value_embed_params,
                'weight_decay': 0.0,
                'lr': learning_rate * value_embed_lr_scale,
                'group_name': 'value_embed',
            })
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_value_embed_params = sum(p.numel() for p in value_embed_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        if value_embed_params:
            print(f"value embedding parameters: {num_value_embed_params:,}, lr scale {value_embed_lr_scale:g}, no decay")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        # The Value Embedding bank is accessed by lookup, not multiplied once per
        # token like a dense model weight. Exclude it from the 6N FLOPs proxy.
        if self.config.value_embeds:
            N -= self.transformer.vte.weight.numel()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the current step
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert the logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# -----------------------------------------------------------------------------
# Muon optimizer (MomentUm Orthogonalized by Newton-Schulz iteration)
# reference: Keller Jordan, https://github.com/KellerJordan/modded-nanogpt
# Muon is used for 2D weight matrices of hidden layers (attention/FFN);
# embeddings, lm_head and 1D params (LayerNorm etc.) should stay on AdamW.

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7) # ensure spectral norm <= 1 before iteration
    if G.size(0) > G.size(1):
        X = X.T # transpose so the Gram matrix A is small
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

@torch.no_grad()
def zeropower_via_newtonschulz5_batched(G, steps=5):
    # batched variant of zeropower_via_newtonschulz5: G is [N, m, n] and every batch
    # element is orthogonalized independently (all shapes in a batch are identical,
    # so the transpose decision is uniform and results are per-matrix equivalent)
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    transposed = G.size(1) > G.size(2)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        # Keep one momentum buffer for each stored parameter. A fused QKV buffer
        # remains [3C, C], but its three row blocks are orthogonalized separately.
        ready = [] # list of (parameter_view, gradient_view, group)
        for group in self.param_groups:
            split_rows = group.get('muon_split_rows', 1)
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                g = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                if group['weight_decay'] > 0:
                    # Decay the stored parameter once, not once per logical block.
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                if split_rows == 1:
                    ready.append((p, g, group))
                    continue
                assert p.ndim == 2 and p.size(0) % split_rows == 0
                rows_per_matrix = p.size(0) // split_rows
                p_views = p.view(split_rows, rows_per_matrix, p.size(1))
                g_views = g.view(split_rows, rows_per_matrix, g.size(1))
                ready.extend((pv, gv, group) for pv, gv in zip(p_views, g_views))

        # Batch equal logical shapes. This batches Q/K/V blocks across all layers.
        groups_by_shape = {}
        for item in ready:
            p_view, g_view, group = item
            key = (tuple(g_view.shape), group['ns_steps'])
            groups_by_shape.setdefault(key, []).append(item)
        for (_, ns_steps), items in groups_by_shape.items():
            batched = torch.stack([g for _, g, _ in items])
            updates = zeropower_via_newtonschulz5_batched(batched, steps=ns_steps)
            for (p_view, _, group), u in zip(items, updates.unbind(0)):
                # Use each logical matrix's aspect ratio. Fused QKV therefore has
                # the same update scale as three original [C, C] parameters.
                aspect_scale = max(1.0, p_view.size(0) / p_view.size(1)) ** 0.5
                p_view.add_(u, alpha=-group['lr'] * aspect_scale)
