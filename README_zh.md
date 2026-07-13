# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**2025 年 11 月更新** nanoGPT 有了一个更新更好的姊妹项目：[nanochat](https://github.com/karpathy/nanochat)。你很可能其实是想找 / 使用 nanochat。nanoGPT（本仓库）已经很旧且已弃用，但我会保留它作为纪念。

---

这是用于训练 / 微调中等规模 GPT 的最简单、最快的仓库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重写版，更侧重「能干活」而非教学。仍在积极开发中，但目前 `train.py` 已能在单台 8×A100 40GB 节点上，用约 4 天训练复现 OpenWebText 上的 GPT-2（124M）。代码本身简洁可读：`train.py` 是约 300 行的样板训练循环，`model.py` 是约 300 行的 GPT 模型定义，并可选择加载 OpenAI 的 GPT-2 权重。就这些。

![repro124m](assets/gpt2_124M_loss.png)

因为代码足够简单，很容易按需魔改、从头训练新模型，或微调预训练检查点（例如当前可用的最大起点是 OpenAI 的 GPT-2 1.3B 模型）。

## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers`：huggingface transformers <3（用于加载 GPT-2 检查点）
- `datasets`：huggingface datasets <3（若需下载并预处理 OpenWebText）
- `tiktoken`：OpenAI 的快速 BPE 实现 <3
- `wandb`：可选日志 <3
- `tqdm`：进度条 <3

## 快速开始

如果你不是深度学习专业人士，只想感受一下魔力、先上手试试，最快的方式是在莎士比亚作品上训练一个字符级 GPT。首先把它下载成单个（1MB）文件，再把原始文本转成一大串整数：

```sh
python data/shakespeare_char/prepare.py
```

这会在该数据目录下生成 `train.bin` 和 `val.bin`。接下来就可以训练你的 GPT 了。模型大小很大程度上取决于你的计算资源：

**我有 GPU**。很好，我们可以用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 配置文件里的设置，快速训练一个小 GPT：

```sh
python train.py config/train_shakespeare_char.py
```

如果你看一下配置，会发现我们在训练一个上下文最长 256 个字符、384 个特征通道、6 层 Transformer（每层 6 个注意力头）的 GPT。在一张 A100 GPU 上，这次训练大约 3 分钟，最佳验证损失为 1.4697。按配置，模型检查点会写入 `--out_dir` 目录 `out-shakespeare-char`。训练结束后，把采样脚本指向该目录即可从最佳模型采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哈哈 `¯\_(ツ)_/¯`。对一个只在 GPU 上训练了 3 分钟的字符级模型来说，还不赖。若改为在该数据集上微调预训练 GPT-2，通常能得到更好结果（见后文微调一节）。

**我只有一台 MacBook**（或其他廉价电脑）。别担心，仍然可以训练 GPT，只是要把规模调小一点。建议安装最新的 PyTorch nightly（[在此选择](https://pytorch.org/get-started/locally/)），当前很可能让代码更高效。即便不用它，一次简单训练也可以像这样：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

这里因为在 CPU 而非 GPU 上跑，必须设置 `--device=cpu`，并关闭 PyTorch 2.0 的 compile（`--compile=False`）。评估时用更噪但更快的估计（`--eval_iters=20`，从 200 降下来），上下文长度只有 64 个字符而非 256，每步 batch size 只有 12 而非 64。我们还会用小得多的 Transformer（4 层、4 头、128 维嵌入），并把迭代次数降到 2000（通常也用 `--lr_decay_iters` 把学习率衰减到大约 max_iters）。因为网络很小，也放宽正则（`--dropout=0.0`）。这大约仍需 ~3 分钟，但损失只有 1.88，样本也更差，不过依然挺好玩：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

会生成类似这样的样本：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

对 CPU 上约 3 分钟的训练来说不错，能看出一点正确的字符风格。若愿意等更久，可以调超参、加大网络、加长上下文（`--block_size`）、加长训练等。

最后，在 Apple Silicon MacBook 且 PyTorch 版本较新时，请加上 `--device=mps`（Metal Performance Shaders 的缩写）；PyTorch 会使用片上 GPU，可**显著**加速训练（2–3 倍），并允许使用更大网络。更多见 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## 复现 GPT-2

更认真的深度学习从业者可能更关心复现 GPT-2 结果。那就开始吧——我们先对数据集做分词，这里用的是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，即 OpenAI（私有）WebText 的开源复现：

```sh
python data/openwebtext/prepare.py
```

这会下载并对 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集做分词。会生成 `train.bin` 和 `val.bin`，以一条序列保存 GPT-2 BPE token id，按原始 uint16 字节存储。然后就可以开训了。要复现 GPT-2（124M），你至少需要一台 8×A100 40GB 节点，并运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这会用 PyTorch Distributed Data Parallel（DDP）跑大约 4 天，损失降到约 ~2.85。直接在 OWT 上评估的 GPT-2 验证损失约为 3.11，但若微调则会进入 ~2.85 区间（看起来存在域差距），使两个模型大致相当。

若你在集群环境，且有幸拥有多台 GPU 节点，可以让 GPU 狂飙，例如跨 2 个节点：

```sh
# 在主节点上运行，示例 IP 为 123.456.123.456：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# 在工作节点上运行：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

建议对互连做基准测试（例如 iperf3）。特别是如果你没有 Infiniband，请在上述启动命令前加上 `NCCL_IB_DISABLE=1`。多节点训练仍能跑，但大概率会**爬行**。默认检查点会定期写入 `--out_dir`。采样只需 `python sample.py`。

最后，单卡训练直接运行 `python train.py` 即可。看看它的所有参数——脚本尽量保持可读、可魔改、透明。你多半会按需调整其中不少变量。

## 基线

OpenAI 的 GPT-2 检查点让我们可以在 openwebtext 上建立一些基线。可如下获取数字：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

并观察到如下 train / val 损失：

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

但需注意：GPT-2 是在（封闭、从未公开的）WebText 上训练的，而 OpenWebText 只是对该数据集的尽力开源复现。因此存在数据集域差距。确实，拿 GPT-2（124M）检查点在 OWT 上直接微调一段时间，损失可降到 ~2.85。这才是更适合作为复现目标的基线。

## 微调

微调与训练并无本质区别，只需确保从预训练模型初始化，并用更小的学习率训练。若想看如何在新文本上微调 GPT，请到 `data/shakespeare` 运行 `prepare.py`，下载 tiny shakespeare 数据集，并用 GPT-2 的 OpenAI BPE 分词器写成 `train.bin` 和 `val.bin`。与 OpenWebText 不同，这只需几秒。微调可以很快，例如单卡只需几分钟。示例微调：

```sh
python train.py config/finetune_shakespeare.py
```

这会加载 `config/finetune_shakespeare.py` 中的配置覆盖（我其实没怎么调过超参）。基本上，用 `init_from` 从 GPT-2 检查点初始化，然后正常训练，只是更短、学习率更小。若显存不够，可尝试减小模型规模（可选 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`），或减小 `block_size`（上下文长度）。最佳检查点（最低验证损失）会在 `out_dir` 目录中，例如按配置文件默认为 `out-shakespeare`。然后可运行 `sample.py --out_dir=out-shakespeare`：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇，GPT，这写到挺黑暗的地方了。配置里的超参我没怎么认真调，欢迎自行尝试！

## 采样 / 推理

用脚本 `sample.py` 可从 OpenAI 发布的预训练 GPT-2 模型采样，也可从你自己训练的模型采样。例如，从最大的可用 `gpt2-xl` 模型采样：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

若要从自己训练的模型采样，用 `--out_dir` 指向相应目录。也可以用文件中的文本作为提示，例如：```python sample.py --start=FILE:prompt.txt```。

## 效率说明

对于简单的模型基准测试与性能分析，`bench.py` 可能有用。它与 `train.py` 训练循环的核心相同，但省略了许多其他复杂性。

注意：代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写时（2022 年 12 月 29 日），nightly 版本已提供 `torch.compile()`。这一行代码带来的提升很明显，例如把迭代时间从约 250ms / iter 降到 135ms / iter。干得漂亮，PyTorch 团队！

## 待办

- 研究并加入 FSDP 以替代 DDP
- 在标准评测上评估 zero-shot 困惑度（如 LAMBADA？HELM？等）
- 微调「微调脚本」本身，我觉得超参不太好
- 训练过程中线性增大 batch size 的调度
- 引入其他位置编码（rotary、alibi）
- 我认为应在检查点中把优化器 buffer 与模型参数分开
- 增加更多网络健康相关日志（如梯度裁剪事件、幅值）
- 再做一些更好初始化等方面的研究

## 排障

注意：本仓库默认使用 PyTorch 2.0（即 `torch.compile`）。这还比较新且偏实验，并非所有平台都可用（例如 Windows）。若遇到相关报错，可加上 `--compile=False` 关闭。这会变慢，但至少能跑。

若想了解本仓库、GPT 与语言建模的更多背景，可观看我的 [Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)。若已有一些语言建模基础，[GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) 尤其受欢迎。

更多问题 / 讨论，欢迎到 Discord 的 **#nanoGPT**：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有 nanoGPT 实验由 [Lambda labs](https://lambdalabs.com) 上的 GPU 驱动——我最喜欢的云 GPU 提供商。感谢 Lambda labs 赞助 nanoGPT！
