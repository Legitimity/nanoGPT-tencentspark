# RMSNorm 与 MLP zero-init 结构实验

两组实验均基于当前 dense ReLU² + gated attention + QK-Norm/QK-scale + 1-bank VE + readout backout 基线，只改变一个结构变量。

统一超参数：

```text
单GPU，固定4小时
batch_size = 24
gradient_accumulation_steps = 10
AdamW learning_rate = 3e-3
Muon learning_rate = 0.03
Muon momentum = 0.90
Muon weight decay = 0.01
warmup_iters = 100
min_lr = 1e-6
lr_schedule = cosine
fused_ce = False
embedding_decay = 0.1
value_embed_decay = 0.0
```

## 实验1：Affine pre-RMSNorm

唯一变化：将每层两个pre-LayerNorm和最终`ln_f`替换为带可学习逐通道gain的RMSNorm；当前`bias=False`，因此训练路径没有shift参数。QK内部的parameter-free RMSNorm保持不变。

在GPU 0运行：

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_affine_rmsnorm_speedrun.py \
  --speedrun_mode=True \
  --speedrun_time_limit_seconds=14400 \
  --batch_size=24 \
  --gradient_accumulation_steps=10 \
  --learning_rate=3e-3 \
  --muon_lr=0.03 \
  --muon_momentum=0.9 \
  --muon_weight_decay=0.01 \
  --warmup_iters=100 \
  --min_lr=1e-6 \
  --lr_schedule=cosine \
  --fused_ce=False \
  --embedding_decay=0.1 \
  --value_embed_decay=0.0 \
  > /tmp/exp_affine_rmsnorm_4h.log 2>&1
```

默认W&B run：

```text
crz affine-rmsnorm-lr3e3-muon3e2-mom09-4h
```

## 实验2：MLP输出投影zero-init

唯一变化：每层`mlp.c_proj.weight`初始化为严格零；`attn.c_proj.weight`继续使用现有tiny residual init。模型参数、优化器分组和运行时计算图不变。

在GPU 1运行：

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_mlp_zero_init_speedrun.py \
  --speedrun_mode=True \
  --speedrun_time_limit_seconds=14400 \
  --batch_size=24 \
  --gradient_accumulation_steps=10 \
  --learning_rate=3e-3 \
  --muon_lr=0.03 \
  --muon_momentum=0.9 \
  --muon_weight_decay=0.01 \
  --warmup_iters=100 \
  --min_lr=1e-6 \
  --lr_schedule=cosine \
  --fused_ce=False \
  --embedding_decay=0.1 \
  --value_embed_decay=0.0 \
  > /tmp/exp_mlp_zero_init_4h.log 2>&1
```

默认W&B run：

```text
crz mlp-zero-init-lr3e3-muon3e2-mom09-4h
```

## 运行位置

四个Python文件和`configurator.py`应位于训练仓库根目录，且该目录下应存在`data/openwebtext`。两个命令是独立单GPU进程，不使用DDP或`torchrun`。

## 判定指标

按以下顺序比较当前最佳LR+momentum基线：

1. `val/final_loss_full`；
2. `speedrun/tokens_processed`和completed updates；
3. 稳态step time与MFU；
4. 最后20%训练区间的train loss；
5. `train/grad_norm`及是否频繁触及clip。

RMSNorm实验只有在吞吐不下降且full val改善时保留。MLP zero-init理论上运行时完全等速，若full val没有改善即可直接淘汰。
