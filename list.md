

# 一、效率优化建议

## P0：真正的Fused Linear Cross Entropy

当前每个microbatch会完整生成：

```text
[24, 1024, 50304]
```

共12.36亿个logits：

- BF16约2.30 GiB；
- FP32约4.61 GiB；
- LM head本身约占前向MAC的25.8%。

当前路径是：

```python
logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(...), targets.view(...))
```

因此这是现阶段最值得优化的单点。

需要区分：

- 已经证明为负优化的普通`chunked CE`；
- 真正不物化完整logits的`linear + CE`融合kernel。

候选：

- [Apple Cut Cross Entropy](https://github.com/apple/ml-cross-entropy)
- [Liger FusedLinearCrossEntropy](https://github.com/linkedin/liger-kernel)

两者都是让分类头矩阵乘、logsumexp和CE在分块kernel中完成，并非把已经生成的logits再切片。

**建议门槛：**

- 300–500个稳定updates；
- 中位step time至少改善2%；
- loss和hidden/head梯度与基线对齐；
- 如果你之前测试的已经是CCE/Liger而非普通chunk CE，则跳过此项。

## P1：测试`torch.compile(mode="reduce-overhead")`

目前使用默认模式：

```python
model = torch.compile(model, mode=None)
```

模型形状完全固定，适合测试CUDA Graph路径：

```bash
--compile_mode=reduce-overhead
```

比较时必须统计：

```text
实际编译耗时 + 500 updates总耗时
```

因为正式4小时包含编译时间。

不优先推荐`max-autotune`：它可能提高稳态速度，但额外编译时间在4小时预算内未必能摊平。除非Inductor cache允许在正式实验间复用。

## P2：ReLU²内核优化

当前FFN写法：

```python
x = F.relu(x) ** 2
```

第一步可改为：

```python
x = F.relu(x).square()
```

但在`torch.compile`下，这个语法改动本身未必产生可测收益。

真正值得测试的是modded-nanogpt的Fused ReLU² MLP思路：

```text
Linear → ReLU² → Linear
```

前后向使用Triton专用kernel，减少激活物化和HBM往返。需要确认kernel走SM80/BF16路径，不能包含H100专属TMA、WGMMA或FP8。

预期端到端收益更保守地看约0–3%。

## P3：优化Muon step

当前Muon已经：

- 按shape批量堆叠；
- 批量Newton–Schulz；
- QKVG分块正交化。

剩余问题是每步仍有Python遍历、`stack/unbind`和逐参数`add_`。

建议先profile以下区间：

```text
forward
backward
grad clip
Muon.step
AdamW.step
data preparation
```

只有优化器占step时间超过5%，才值得：

- 单独编译optimizer step；
- 缓存stack buffer；
- 使用参数bank；
- 融合最后的参数更新。

由于当前每10个microsteps才执行一次优化器，优化器融合的端到端收益可能只有0–2%。

## P4：只验证SDPA后端，不急着换FlashAttention

当前形状：

```text
A100
BF16
head_dim=64
causal
dropout=0
```

PyTorch SDPA大概率已经走FlashAttention后端。应先强制：

```python
SDPBackend.FLASH_ATTENTION
```

或通过Profiler查看kernel，确认没有回退到math backend。

- FlashAttention-2支持A100；
- FlashAttention-3和FP8 Tensor Core优化针对H100，不要投入。

