# OpenWebText GPT 实验记录

> 最后更新：2026-07-30 13:11（GMT+8）  
> W&B 项目：https://wandb.ai/jqh333/owt/runs/  
> 各阶段基线代码位于本目录的 `baseline1/`、`baseline2/`、`baseline3/`、`baseline4/`。

## 1. 阅读与比较口径

### 1.1 正式指标

从实验 20 开始，正式结果统一以单卡固定墙钟 Speedrun 的：

```text
val/final_loss_full
```

作为主要指标，即计时训练结束后，对完整验证集做顺序full pass得到的loss。数值越低越好。

中途上传的 `val/loss` 只覆盖少量随机batch，不能代替full validation；运行中的 `train/lm_loss` 只能用于观察趋势，不能提前宣布最终优胜者。

### 1.2 有效Speedrun判定

一个run只有同时满足以下条件，才可与4小时结果直接比较：

```text
speedrun_mode = True
speedrun_time_limit_seconds = 14400
speedrun/termination_reason = time_limit
```

仅仅在config中出现 `speedrun_time_limit_seconds=14400` 不代表真正启用了Speedrun。部分后期 `hp-exp*` run实际是 `speedrun_mode=False`、`max_iters=5000` 的普通训练，运行约1.6–2小时，不能用于4小时排名。

### 1.3 当前Speedrun统一硬件语义

双卡机器上的两张GPU被当作两台独立单卡实验设备使用，不使用DDP。典型配置为：

```text
单GPU、单训练进程
batch_size = 24
gradient_accumulation_steps = 8或10
block_size = 1024
speedrun_time_limit_seconds = 14400
fused_ce = False
```

---

## 2. 当前结论摘要

### 2.1 当前最佳已完成结果

当前最低full validation loss为：

- Run [`ixd9wd56`](https://wandb.ai/jqh333/owt/runs/ixd9wd56)
- `val/final_loss_full = 3.005883`
- 处理tokens约 `2.496B`，完成 `12695` updates
- `speedrun/termination_reason = time_limit`

对应配置：

```text
结构：baseline4，LayerNorm
batch_size = 24
gradient_accumulation_steps = 8
AdamW learning_rate = 0.003
Muon learning_rate = 0.03
Muon momentum = 0.90
Muon weight decay = 0.01
weight_decay = 0.1
embedding_decay = 0.0
value_embed_decay = 0.0
lr_schedule = wsd
wsd_decay_frac = 0.30
wsd_decay_style = linear
warmup_iters = 100
min_lr = 1e-6
```

它比此前最佳cosine run `jqz2tfo5 = 3.012101`降低约`0.006218`。在已完成的WSD linear .30候选中，实验46为`3.008401`，实验48为`3.017362`，实验47的`.003/.03`结果最低；但实验46的W&B配置未记录`embedding_decay`字段，实验48使用`embedding_decay=.1`，因此三者不是严格的单变量peak-LR扫描。

### 2.2 已确认的主要结论

1. **Dense FFN是A100固定时间下的正确主线。** Dense版本MFU约50%，MoE约34%；MoE在样本效率上的潜在收益无法抵消吞吐下降。
2. **QK-Norm + 可学习per-head scale有效。** 相比baseline1约降低0.02。
3. **1-bank Value Embedding有小幅质量收益，速度影响很小。** 早期约降低0.01；固定时间无VE消融也更差。
4. **Tiny residual init、ReLU²和gated attention均有正收益。** 其中ReLU²是结构改动中收益最大的一项。
5. **Readout backout在Dense FFN上的单run点估计边际很小。** `3.03697 → 3.03521`，约改善0.00175；没有多seed重复，统计未定。可保留，但不是主要增益来源。
6. **Embedding shortcut无效。** 固定时间下比对应基线更差。
7. **Muon momentum=0.90稳定优于0.95。** 在两组对照中均有正向信号。
8. **有效batch的最优区域靠近accumulation=8。** accum10与accum8均可，accum6明显变差。
9. **Cosine的局部最佳仍在AdamW约0.0023、Muon约0.026。** 向上提高到`.00250/.028`或只把Muon提高到`.029`均未胜过原cosine最佳。
10. **WSD linear、decay_frac=0.30成为当前最优调度。** 在accum8、momentum .9、embedding decay 0下，peak LR `.003/.03`取得`3.005883`，明显优于此前最佳cosine的`3.012101`。其他WSD候选分别为`3.008401`和`3.017362`，但embedding-decay记录不完全一致，不能视为纯LR消融。
11. **Affine RMSNorm单次匹配run与LayerNorm基本打平。** 吞吐点估计快约0.14%，full val高0.00046；没有多seed重复，统计未定，因此默认仍保留LayerNorm。

---

## 3. Baseline演进

| Baseline | 来源实验 | 核心配置 | 后续用途 |
|---|---:|---|---|
| baseline0 | 1 | nanoGPT GPT-2 124M | 初始训练recipe搜索 |
| baseline1 | 6 | microbatch 6 + Muon | 注意力和VE实验 |
| baseline2 | 10 | baseline1 + 有参QK-Norm + 1-bank VE | 初始化与激活实验 |
| baseline3 | 13 | baseline2 + tiny init + ReLU² + gated attention | Dense/MoE结构比较 |
| baseline4 | 27 | baseline3 + final readout backout + Speedrun | 最终Dense调参与消融 |
| 当前best | 47 | baseline4 + accum8 + `.003/.03` + momentum .9 + WSD linear .30 | 最终推荐 |

---

## 4. 详细实验记录

## 阶段A：baseline0训练recipe（实验1–6）

此阶段基线为原始nanoGPT GPT-2 124M，主要搜索LR、scheduler、microbatch和优化器。

1. [`lj4lkgge`](https://wandb.ai/jqh333/owt/runs/lj4lkgge)：baseline0。
2. [`ah4zq4ft`](https://wandb.ai/jqh333/owt/runs/ah4zq4ft)：`lr=8e-4, min_lr=0`。结果优于实验1。
3. [`z9bqiun4`](https://wandb.ai/jqh333/owt/runs/z9bqiun4)：`lr=8e-4, min_lr=1e-6`。优于实验1，略差于实验2。
4. [`r8a8iga8`](https://wandb.ai/jqh333/owt/runs/r8a8iga8)：WSD linear，`decay_frac=0.2, lr=8e-4, min_lr=1e-6`。相比实验1、2均有显著提升。
5. [`83yfjs5o`](https://wandb.ai/jqh333/owt/runs/83yfjs5o)：`micro_batch_size=6`。训练略慢，但质量明显提升。
6. [`ksxkbtcw`](https://wandb.ai/jqh333/owt/runs/ksxkbtcw)：microbatch 6 + Muon，`lr=1.8e-3, min_lr=1e-6`，未沿用实验4的WSD。成为baseline1。

阶段结论：Muon是关键改进；WSD存在正信号，但其最优参数依赖模型和LR，后续需要重新消融。

## 阶段B：QK-Norm、head数与Value Embedding（实验7–10）

7. [`57gdk0wg`](https://wandb.ai/jqh333/owt/runs/57gdk0wg)：baseline1 + 无参QK-Norm。曲线与baseline1几乎完全相同，提前终止。
8. [`y9xv5vqv`](https://wandb.ai/jqh333/owt/runs/y9xv5vqv)：有参QK-Norm，即对每层每个head加入可学习logit scale（等价于逆温度）。相比baseline1降低约0.02。
9. [`6dmf2vk4`](https://wandb.ai/jqh333/owt/runs/6dmf2vk4)：实验8基础上改为 `n_head=6`。与实验8无明显区别。
10. [`suttsoz3`](https://wandb.ai/jqh333/owt/runs/suttsoz3)：实验8 + 1-bank Value Embedding。效率几乎不受影响，val loss进一步降低约0.01，成为baseline2。

1-bank VE使用一个共享token表，仅注入最后4层：

$$
\widetilde V_i=V_i+\alpha_{i,h}E^V[t],\qquad i=8,9,10,11.
$$

相关工程实现的演进（不是实验10的单独变量）：

- baseline1沿用nanoGPT的单个`[C,3C]` fused QKV投影；
- baseline2为让Muon独立正交化三个投影，暂时拆成独立`wq/wk/wv`；
- baseline3重新合成`[C,3C]` QKV投影，同时保留独立`[C,C]` gate投影，并在Muon中按逻辑row blocks处理；
- Speedrun baseline4再将Q/K/V/gate拼成一个`[C,4C]`投影，数学保持不变但减少一次GEMM launch；
- Muon按逻辑Q/K/V/gate row blocks分别正交化，并按矩阵shape批量执行迭代。

## 阶段C：初始化、激活与gated attention（实验11–13）

11. [`hmrcprxq`](https://wandb.ai/jqh333/owt/runs/hmrcprxq)：baseline2的所有residual projection初始化标准差缩小到理论值的0.1倍。val loss降低约0.01。
12. [`1te73ecr`](https://wandb.ai/jqh333/owt/runs/1te73ecr)：实验11基础上将Dense FFN激活改为ReLU²。val loss降低约0.04。
13. [`f169p0x3`](https://wandb.ai/jqh333/owt/runs/f169p0x3)：实验12基础上增加SDPA输出后的elementwise sigmoid gated attention。val loss再降低约0.01，成为baseline3。

阶段结论：ReLU²是最强的结构增益；tiny residual init与gated attention也稳定有效。

## 阶段D：MoE探索（实验14–19）

14. [`eo34xc4v`](https://wandb.ai/jqh333/owt/runs/eo34xc4v)：baseline2，4专家top-1，`expert_dim=3072`，GELU。val loss到3.01级别，但耗时大幅超过4小时，结果不能与固定时间Dense模型比较。
15. [`5h5batk2`](https://wandb.ai/jqh333/owt/runs/5h5batk2)：baseline3 + MoE，`expert_dim=2048`，ReLU²。质量保持3.01级别，耗时仍超过4小时。
16. [`livno3r7`](https://wandb.ai/jqh333/owt/runs/livno3r7)：实验15的GELU对照。与ReLU²无明显区别。
17. [`c0zl4sog`](https://wandb.ai/jqh333/owt/runs/c0zl4sog)：实验15的`expert_dim=1536`版本。val loss升高约0.02，速度没有显著提高。
18. [`pt5lgead`](https://wandb.ai/jqh333/owt/runs/pt5lgead)：实验17移除VE。val loss升高约0.01，速度无明显提升。
19. [`dhnrre3w`](https://wandb.ai/jqh333/owt/runs/dhnrre3w)：实验15改为等参数SwiGLU。比实验15稍慢，val loss改善约0.01至3.00级别。

阶段结论：MoE样本效率并非完全无效，但A100上sort、dispatch和grouped GEMM使MFU从Dense约50%降到约34%，固定4小时目标下整体失败；后续停止MoE主线。

## 阶段E：引入固定4小时Speedrun（实验20–27）

Speedrun模式以墙钟时间为唯一停止条件；warmup后根据实测速率估计LR调度horizon，并保留约1% headroom；结束后执行完整验证集full pass。

20. [`xyk105kx`](https://wandb.ai/jqh333/owt/runs/xyk105kx)：baseline3 Dense单卡4小时。`full val=3.036966`。
21. [`2ita4mt5`](https://wandb.ai/jqh333/owt/runs/2ita4mt5)：MoE `expert_dim=1536` Speedrun。`3.054115`。
22. [`gl69qxls`](https://wandb.ai/jqh333/owt/runs/gl69qxls)：实验21改为等参数SwiGLU。`3.051846`。
23. [`nodgryim`](https://wandb.ai/jqh333/owt/runs/nodgryim)：实验22改为WSD。`3.044938`。
24. [`qkjsgjgn`](https://wandb.ai/jqh333/owt/runs/qkjsgjgn)：实验21加入final readout backout。`3.048393`。
25. [`36yatwrq`](https://wandb.ai/jqh333/owt/runs/36yatwrq)：实验21加入identity-init embedding shortcut。`3.056808`。
26. [`2oy1srre`](https://wandb.ai/jqh333/owt/runs/2oy1srre)：实验22改为efficient LatentMoE。中途曲线明显变差，停止/崩溃，无最终full val。
27. [`02bxhv7u`](https://wandb.ai/jqh333/owt/runs/02bxhv7u)：实验20的Dense模型加入final readout backout。`3.035213`，成为baseline4。

阶段结论：Dense继续明显领先；readout backout在Dense单次对照中点估计改善约0.00175，但没有多seed重复；embedding shortcut固定时间为负。

## 阶段F：baseline4 Dense消融（实验28–34）

28. [`oycn80w3`](https://wandb.ai/jqh333/owt/runs/oycn80w3)：baseline4关闭wte/wpe embedding weight decay。`3.036915`，单独关闭没有改善。
29. [`44pjeowh`](https://wandb.ai/jqh333/owt/runs/44pjeowh)：实验28基础上完全删除1-bank VE。`3.041136`，进一步变差；VE在固定时间下仍有实际价值。
30. [`ih5g5g12`](https://wandb.ai/jqh333/owt/runs/ih5g5g12)：baseline4将Dense ReLU² FFN改为等参数SwiGLU。`3.040597`，固定时间为负。
31. [`ipqdqw4s`](https://wandb.ai/jqh333/owt/runs/ipqdqw4s)：实验30将gradient accumulation从10降到8。`3.034069`，比实验30明显改善。
32. [`acrgnm23`](https://wandb.ai/jqh333/owt/runs/acrgnm23)：实验30将gradient accumulation降到6，并调整LR/warmup。`3.042328`，说明更新过于频繁或对应LR不合适。
33. [`xluqrqz6`](https://wandb.ai/jqh333/owt/runs/xluqrqz6)：baseline4将Muon momentum从0.95改为0.90。`3.030237`，明显改善。
34. [`72tuhk4p`](https://wandb.ai/jqh333/owt/runs/72tuhk4p)：baseline4将AdamW/Muon LR提高到 `.003/.03`，momentum仍为0.95。`3.024343`，说明原LR明显偏低。

阶段结论：momentum 0.90和更高LR均有效；accum8值得继续搜索；Dense SwiGLU和embedding no-decay单独均未胜出。

## 阶段G：有效单卡4小时超参搜索（实验35–43）

以下均为确认启用 `speedrun_mode=True` 的完整4小时结果。

35. [`8wyxo79x`](https://wandb.ai/jqh333/owt/runs/8wyxo79x)：accum8，embedding decay 0，`.0018/.02`，momentum .9，cosine。`3.023985`。
36. [`ff86udlz`](https://wandb.ai/jqh333/owt/runs/ff86udlz)：accum8，embedding decay 0，`.0021/.02`，momentum .9，cosine。`3.014872`。
37. [`jqz2tfo5`](https://wandb.ai/jqh333/owt/runs/jqz2tfo5)：accum8，embedding decay 0，`.00234/.026`，momentum .9，cosine。`3.012101`，当时的最佳cosine结果。
38. [`c0kb4ija`](https://wandb.ai/jqh333/owt/runs/c0kb4ija)：旧代码语义下，accum10，AdamW LR约`.00306`、Muon LR `.02`、momentum .95、cosine。`3.019023`。
39. [`jllt2aq1`](https://wandb.ai/jqh333/owt/runs/jllt2aq1)：accum10，embedding decay .1，`.003/.03`，momentum .9，cosine。`3.013418`。
40. [`y4jdzfcz`](https://wandb.ai/jqh333/owt/runs/y4jdzfcz)：accum10，旧embedding decay语义，`.003/.03`，momentum .95，WSD linear，`decay_frac=.30`。`3.016538`。
41. [`78vxa0cs`](https://wandb.ai/jqh333/owt/runs/78vxa0cs)：accum10，embedding decay .1，`.003/.03`，momentum .95，WSD cosine，`decay_frac=.20`。`3.032438`。
42. [`wir3zra1`](https://wandb.ai/jqh333/owt/runs/wir3zra1)：accum10，旧embedding decay语义，`.003/.03`，momentum .95，WSD linear，`decay_frac=.10`。`3.056295`。
43. [`bflm0z0y`](https://wandb.ai/jqh333/owt/runs/bflm0z0y)：Affine RMSNorm，accum10，embedding decay .1，`.003/.03`，momentum .9，cosine。`3.013878`；约2.509B tokens。匹配LayerNorm实验39为`3.013418`、约2.506B tokens。单次run中RMSNorm快约0.14%、full val高0.00046；无多seed重复，只能视为点估计平局偏负。

截至实验43的有效结果排名前五（后续已被阶段I的WSD结果刷新）：

| 排名 | Run | Full val | 关键配置 |
|---:|---|---:|---|
| 1 | `jqz2tfo5` | **3.012101** | accum8，`.00234/.026`，mom .9，cosine |
| 2 | `jllt2aq1` | 3.013418 | accum10，`.003/.03`，mom .9，cosine |
| 3 | `bflm0z0y` | 3.013878 | Affine RMSNorm，配置同上 |
| 4 | `ff86udlz` | 3.014872 | accum8，`.0021/.02`，mom .9，cosine |
| 5 | `y4jdzfcz` | 3.016538 | WSD linear .30，`.003/.03`，mom .95 |

阶段结论：最优区域已经收缩到LayerNorm、accum8、momentum .9、AdamW LR约0.0023、Muon LR约0.026；当前单run证据没有显示RMSNorm替换LayerNorm的必要性。

## 阶段H：非4小时或失败run，不纳入排名

下列run虽然名称或config中包含14400秒，但实际 `speedrun_mode=False`，仅运行5000 updates，不能据此判断相应超参数优劣：

| Run | 主要配置 | 状态/说明 |
|---|---|---|
| [`oliycj8s`](https://wandb.ai/jqh333/owt/runs/oliycj8s) | accum8，名义`.0018/.03` | 普通5000-step |
| [`yrjx840w`](https://wandb.ai/jqh333/owt/runs/yrjx840w) | Muon LR .035 | 普通5000-step |
| [`mtp5qxof`](https://wandb.ai/jqh333/owt/runs/mtp5qxof) | 低peak WSD `.0024/.024` | 普通5000-step |
| [`8qrp7cr1`](https://wandb.ai/jqh333/owt/runs/8qrp7cr1) | accum8 `.002/.02` | 普通5000-step |
| [`p9ra8law`](https://wandb.ai/jqh333/owt/runs/p9ra8law) | Muon LR .025 | 普通5000-step |
| [`16zn6p0l`](https://wandb.ai/jqh333/owt/runs/16zn6p0l) | `.0018/.018` | 普通5000-step |
| [`vcz9zi5g`](https://wandb.ai/jqh333/owt/runs/vcz9zi5g) | Muon WD .005 | 普通5000-step |
| [`tuiamwzb`](https://wandb.ai/jqh333/owt/runs/tuiamwzb) | AdamW LR .0036 | 普通5000-step |
| [`7y4bh9qm`](https://wandb.ai/jqh333/owt/runs/7y4bh9qm) | WSD cosine .20 | 普通训练，非Speedrun |

另外：

- [`yijtfdv1`](https://wandb.ai/jqh333/owt/runs/yijtfdv1)：WSD linear .20，启用Speedrun但早期崩溃；
- [`293h9sp6`](https://wandb.ai/jqh333/owt/runs/293h9sp6)：启动失败；
- [`v9zi62uu`](https://wandb.ai/jqh333/owt/runs/v9zi62uu)：启动/运行崩溃。

这些run只用于排查启动命令和配置注入问题，不进入质量结论。

## 阶段I：最终调度与LR候选（实验44–50）

> 以下均按单卡4小时口径判断。WSD在前70%训练中保持peak LR，因此其中段train loss不能与cosine直接判胜；本节以最终full validation为准。

44. [`n56pdz7d`](https://wandb.ai/jqh333/owt/runs/n56pdz7d)：cosine，accum8，embedding decay 0，`.00250/.028`，momentum .9。`3.014895`；联合提高两侧LR未胜过实验37。
45. [`z0nhvru9`](https://wandb.ai/jqh333/owt/runs/z0nhvru9)：cosine，accum8，embedding decay 0，`.00234/.029`，momentum .9。`3.013985`；只提高Muon LR也未胜过实验37。
46. [`eorictw2`](https://wandb.ai/jqh333/owt/runs/eorictw2)：WSD linear .30，accum8，`.00234/.026`，momentum .9。`3.008401`，约2.511B tokens；首次明确超过cosine历史最佳。注意W&B config未记录`embedding_decay`字段，不能据此做严格的embedding-decay归因。
47. [`ixd9wd56`](https://wandb.ai/jqh333/owt/runs/ixd9wd56)：WSD linear .30，accum8，embedding decay 0，`.003/.03`，momentum .9。**`3.005883`**，约2.496B tokens、12695 updates，成为当前最佳。
48. [`qd5zuqua`](https://wandb.ai/jqh333/owt/runs/qd5zuqua)：WSD linear .30，accum8，embedding decay .1，`.0036/.04`，momentum .9。`3.017362`；明显弱于实验47，但它同时改变了peak LR和embedding decay，不能把差异全部归因于LR过高。
49. [`l2uf4rxg`](https://wandb.ai/jqh333/owt/runs/l2uf4rxg)：cosine，accum8，embedding decay .1，`.0018/.03`，momentum .9。`3.018754`；低AdamW LR配高Muon LR无优势。
50. [`y81gvwwe`](https://wandb.ai/jqh333/owt/runs/y81gvwwe)：cosine，accum8，embedding decay .1，`.0018/.018`，momentum .9。约step 4030后crashed，无最终full val。

阶段结论：

- WSD linear .30在当前模型和4小时预算上具有真实优势；
- 在已完成WSD候选中，实验47比实验46低约`0.002518`、比实验48低约`0.011478`；但46缺少embedding-decay记录、48使用`.1`而47使用`0`，不能把这些差值解释为纯peak-LR主效应；
- 当前已观测到的最佳组合是AdamW LR `.003`、Muon LR `.03`、embedding decay 0；更精确的局部最优仍需严格同口径复现确认；
- cosine局部上探和单独提高Muon均失败，因此最终推荐从实验37切换到实验47。

当前所有已完成有效Speedrun的前五名：

| 排名 | 实验 / Run | Full val | 关键配置 |
|---:|---|---:|---|
| 1 | 47 / `ixd9wd56` | **3.005883** | accum8，`.003/.03`，mom .9，WSD linear .30 |
| 2 | 46 / `eorictw2` | 3.008401 | accum8，`.00234/.026`，mom .9，WSD linear .30 |
| 3 | 37 / `jqz2tfo5` | 3.012101 | accum8，`.00234/.026`，mom .9，cosine |
| 4 | 39 / `jllt2aq1` | 3.013418 | accum10，`.003/.03`，mom .9，cosine |
| 5 | 43 / `bflm0z0y` | 3.013878 | Affine RMSNorm，配置同实验39 |

---

## 5. 方向性总结

### 5.1 结构方向

| 方向 | 结论 |
|---|---|
| 有参QK-Norm | 明确正收益，保留 |
| n_head 12→6 | 无明显收益 |
| 1-bank VE | 小幅正收益，速度影响很小，保留 |
| Tiny residual init | 正收益，保留 |
| ReLU² | 明确正收益，保留 |
| Gated attention | 小幅正收益，保留 |
| Readout backout | Dense单run点估计边际很小，统计未定；可保留但不再重点投入 |
| Embedding shortcut | 固定时间为负，淘汰 |
| Dense SwiGLU | 固定时间未胜过ReLU²，淘汰 |
| Affine RMSNorm | 单次匹配run与LayerNorm近似打平，统计未定；默认保留LayerNorm |
| MoE/LatentMoE | A100固定时间吞吐损失过大，淘汰 |

### 5.2 优化器和调度方向

| 方向 | 结论 |
|---|---|
| Muon momentum .90 | 优于.95，保留 |
| accum10→8 | 有效；accum6过度 |
| AdamW/Muon LR | WSD下当前最佳peak为`.003/.03`；cosine局部最佳仍约`.00234/.026` |
| 更高Muon LR | cosine中单独提高到`.029`未改善；WSD需要与AdamW peak共同匹配 |
| WSD linear .10 | cooldown过短，淘汰 |
| WSD cosine .20 | 未胜，淘汰 |
| WSD linear .30 | 当前最佳调度，实验47达到`3.005883` |
| embedding decay 0 | 单独消融未改善，但当前最佳组合使用0；不能归因成独立正收益 |

---

## 6. 当前推荐与剩余候选

### 6.1 当前默认推荐

在所有已完成、有效的单卡4小时run中，默认采用实验47：

```text
LayerNorm
batch_size = 24
gradient_accumulation_steps = 8
AdamW LR = 0.003
Muon LR = 0.03
Muon momentum = 0.90
Muon weight decay = 0.01
weight_decay = 0.1
embedding_decay = 0.0
value_embed_decay = 0.0
warmup_iters = 100
min_lr = 1e-6
lr_schedule = wsd
wsd_decay_frac = 0.30
wsd_decay_style = linear
fused_ce = False
speedrun_mode = True
speedrun_time_limit_seconds = 14400
```

### 6.2 尚在运行的局部补点

截至本次更新，另有两条与当前最优区域相邻的run仍在运行：

1. [`j7kcjlj9`](https://wandb.ai/jqh333/owt/runs/j7kcjlj9)：WSD linear .30，AdamW `.00267`、Muon `.028`、momentum `.9`、accum8；用于检查`.00234/.026`和`.003/.03`之间是否存在更优中点。
2. [`rzlp29xg`](https://wandb.ai/jqh333/owt/runs/rzlp29xg)：cosine，AdamW `.00234`、Muon `.024`、momentum `.9`、accum8；用于补齐cosine的低Muon侧。

在它们完成前，正式推荐保持实验47。最终仍只比较`val/final_loss_full`；若中间WSD点未低于`3.005883`，则不再需要扩大搜索范围。
