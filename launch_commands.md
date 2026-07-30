# 最终超参数 Speedrun 实验

## 结论依据

本机已经能够通过 W&B API 访问 `jqh333/owt` 中的实验。设计依据：

- `02bxhv7u`：AdamW LR `1.8e-3`、Muon LR `0.02`、momentum `0.95`、cosine，full val loss `3.03521`。
- `72tuhk4p`：AdamW LR `3e-3`、Muon LR `0.03`，full val loss `3.02434`，是当前最优结果，作为新锚点。
- `xluqrqz6`：Muon momentum `0.90`，full val loss约`3.030`；优于原momentum `0.95`基线，但弱于最佳LR实验，因此保留“最佳LR + momentum 0.90”的组合测试。
- 早期baseline0及MoE实验均出现过WSD正信号，因此在当前dense结构上系统测试cooldown长度与形状。

## 统一单卡条件

所有实验使用相同结构：dense ReLU² FFN、QK-Norm、可学习QK scale、gated attention、1-bank VE、tiny residual init、readout backout。

```text
单GPU、单训练进程
speedrun_time_limit_seconds = 14400
batch_size = 24
gradient_accumulation_steps = 10
block_size = 1024
fused_ce = False
embedding_decay = 0.1
value_embed_decay = 0.0
warmup_iters = 100
min_lr = 1e-6
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
```

不使用`torchrun`，也不让一个实验同时占用两张GPU。双卡机器上的两张卡作为两台独立实验设备使用：在两个终端分别设置`CUDA_VISIBLE_DEVICES=0`和`CUDA_VISIBLE_DEVICES=1`，每张卡同时只运行一组实验。这样每组结果都与历史单卡4小时speedrun直接可比。

以下命令应在训练仓库根目录执行；该目录须包含`configurator.py`和`data/openwebtext`。将本`code`目录复制或链接到训练仓库根目录，或把`code/train_expN.py`换成实际路径。

## 优先级与变量

| 优先级 | 文件 | AdamW LR | Muon LR | Momentum | Muon WD | Schedule | WSD frac/style | 目的 |
|---:|---|---:|---:|---:|---:|---|---|---|
| 1 | `train_exp1.py` | 0.0030 | 0.030 | 0.95 | 0.010 | cosine | — | 单卡4小时复现当前最佳LR |
| 2 | `train_exp2.py` | 0.0030 | 0.030 | 0.95 | 0.010 | WSD | 0.20 / linear | 首选WSD候选 |
| 3 | `train_exp3.py` | 0.0030 | 0.030 | 0.90 | 0.010 | cosine | — | 最佳LR与momentum 0.90组合 |
| 4 | `train_exp4.py` | 0.0030 | 0.030 | 0.95 | 0.010 | WSD | 0.30 / linear | 更长cooldown |
| 5 | `train_exp5.py` | 0.0030 | 0.030 | 0.95 | 0.010 | WSD | 0.10 / linear | 更短cooldown |
| 6 | `train_exp6.py` | 0.0030 | 0.030 | 0.95 | 0.010 | WSD | 0.20 / cosine | 隔离cooldown形状 |
| 7 | `train_exp7.py` | 0.0024 | 0.030 | 0.95 | 0.010 | cosine | — | 单独降低AdamW侧LR |
| 8 | `train_exp8.py` | 0.0036 | 0.030 | 0.95 | 0.010 | cosine | — | 单独提高AdamW侧LR |
| 9 | `train_exp9.py` | 0.0030 | 0.025 | 0.95 | 0.010 | cosine | — | 单独降低Muon侧LR |
| 10 | `train_exp10.py` | 0.0030 | 0.035 | 0.95 | 0.010 | cosine | — | 单独提高Muon侧LR |
| 11 | `train_exp11.py` | 0.0030 | 0.030 | 0.95 | 0.005 | cosine | — | 测试较弱Muon weight decay |
| 12 | `train_exp12.py` | 0.0024 | 0.024 | 0.95 | 0.010 | WSD | 0.20 / linear | 测试WSD是否偏好较低peak LR |

## 启动顺序

建议每轮并行两个相邻优先级实验：GPU 0运行奇数编号，GPU 1运行偶数编号；一轮结束后再启动下一轮。

### 第一轮：Exp 1 / Exp 2

GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp1.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp1.log 2>&1
```

GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp2.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp2.log 2>&1
```

### 第二轮：Exp 3 / Exp 4

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp3.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp3.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp4.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp4.log 2>&1
```

### 第三轮：Exp 5 / Exp 6

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp5.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp5.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp6.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp6.log 2>&1
```

### 第四轮：Exp 7 / Exp 8

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp7.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp7.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp8.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp8.log 2>&1
```

### 第五轮：Exp 9 / Exp 10

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp9.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp9.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp10.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp10.log 2>&1
```

### 第六轮：Exp 11 / Exp 12

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/train_exp11.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp11.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=1 python3 code/train_exp12.py \
  --speedrun_mode=True --speedrun_time_limit_seconds=14400 \
  --batch_size=24 --gradient_accumulation_steps=10 \
  --fused_ce=False --embedding_decay=0.1 --value_embed_decay=0.0 \
  --warmup_iters=100 --min_lr=1e-6 \
  > /tmp/hp_exp12.log 2>&1
```

## WSD解释与判定

当前实现：

```text
warmup: 0 → peak LR，100 updates
stable: peak LR保持不变
cooldown: 在估计max_iters的最后frac比例内衰减到min_lr
```

- `frac=0.1`：最后约10% updates退火，最长高LR稳定区；
- `frac=0.2`：首选折中，也是baseline0已有正信号的设置；
- `frac=0.3`：更早进入退火，对较高peak LR更保守；
- linear在cooldown早期比cosine更快降LR；cosine在两端更平滑。

Speedrun在warmup后半段测速并估计`max_iters`，附加1% headroom；墙钟时间仍是唯一停止条件。最终应比较：

1. `val/final_loss_full`；
2. `speedrun/tokens_processed`与completed updates；
3. 最后20%训练区间的train loss均值；
4. `train/grad_norm`是否频繁触及clip；
5. WSD的最终schedule progress与结束LR。

资源有限时，优先完成前三轮：`Exp 1–6`。
