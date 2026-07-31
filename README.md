本仓库基于nanoGPT，为2026年腾讯星火计划中进行的speedrun工程。

# GPT结构迭代技术报告

> 本文与 [`experiment.md`](./experiment.md) 的实验编号一一对应，重点记录模型结构如何从nanoGPT GPT-2逐步演化到当前约170M参数的Dense Speedrun模型。  
> 训练超参数、W&B状态及完整loss结果以 `experiment.md` 为准；本文侧重数学定义、初始化、代码实现、参数/计算代价与结构结论。

## 1. 符号和最终模型概览

统一记号：

- 层数：$L=12$
- hidden width：$C=768$
- attention heads：$H=12$
- head dimension：$d_h=C/H=64$
- sequence length：$T=1024$
- vocabulary：$V=50304$
- token hidden states：$X\in\mathbb{R}^{B\times T\times C}$

当前经过实验确认的主线结构约有 **170.09M** 参数（含position embedding），主要由以下部分组成：

```text
Token embedding / tied LM head: 50304 × 768
Position embedding:             1024 × 768
1 shared Value Embedding bank:  50304 × 768
12 × Pre-LayerNorm Transformer blocks
  - fused Q/K/V/attention-gate projection: 768 → 4×768
  - QK RMS normalization + learned per-head scale
  - SDPA output elementwise sigmoid gate
  - Dense ReLU² FFN: 768 → 3072 → 768
  - tiny residual-projection initialization
Final LayerNorm
Identity-init final readout backout from block 7
```

当前最终block可概括为：

$$
\begin{aligned}
U_l &= \operatorname{LN}_1(X_l),\\
A_l &= \operatorname{GatedAttention}(U_l,E^V),\\
X'_l &= X_l + A_l,\\
M_l &= W_{2,l}\left[\operatorname{ReLU}(W_{1,l}\operatorname{LN}_2(X'_l))\right]^2,\\
X_{l+1} &= X'_l + M_l.
\end{aligned}
$$

其中Value Embedding只注入最后4层；最终LM head输入为：

$$
Z=a\,\operatorname{LN}_f(X_{12})+b\,\operatorname{LN}_f(X_8),
\qquad [a,b]_{\text{init}}=[1,0].
$$

---

## 2. Baseline0：原始nanoGPT GPT-2（实验1–6）

### 2.1 标准Pre-LN block

原始模型来自nanoGPT风格GPT-2，代码对应 [`baseline1/model.py`](./baseline1/model.py)。注意：`baseline1`目录保存的是实验6之后使用的模型主体，但其block结构仍是标准nanoGPT block。

数学形式：

$$
\begin{aligned}
X'_l &= X_l+\operatorname{MHA}(\operatorname{LN}_1(X_l)),\\
X_{l+1} &= X'_l+\operatorname{MLP}(\operatorname{LN}_2(X'_l)).
\end{aligned}
$$

标准attention：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\text{causal}}\right)V.
$$

标准Dense FFN：

$$
\operatorname{MLP}(x)=W_2\operatorname{GELU}(W_1x),
\qquad W_1:C\to4C,\ W_2:4C\to C.
$$

关键代码：

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)       # 768 -> 3072
        x = self.gelu(x)
        x = self.c_proj(x)     # 3072 -> 768
        return self.dropout(x)
```

### 2.2 原始初始化

普通Linear和Embedding：

$$
W\sim\mathcal N(0,0.02^2).
$$

Attention和MLP的residual output projection使用GPT-2缩放初始化：

$$
\sigma_{\text{resid}}
=\frac{0.02}{\sqrt{2L}}
=\frac{0.02}{\sqrt{24}}
\approx0.004082.
$$

```python
if pn.endswith('c_proj.weight'):
    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

### 2.3 实验1–6改变了什么

实验1–6主要改变LR、scheduler、microbatch和优化器，没有改变前向结构。实验6引入Muon后成为baseline1。

Muon不是模型结构，但它影响后续结构设计：hidden-layer二维矩阵进入Muon，Embedding、LayerNorm、bias和其他1D参数继续使用AdamW。

---

## 3. QK-Norm与可学习attention logit scale（实验7–9）

代码主体见 [`baseline2/model_ve.py`](./baseline2/model_ve.py)。

### 3.1 实验7：parameter-free QK-Norm

对每个token、每个head的Q/K向量做RMS normalization：

$$
\widehat q_{b,h,t}
=\frac{q_{b,h,t}}
{\sqrt{\frac1{d_h}\sum_jq_{b,h,t,j}^2+\epsilon}},
$$

$$
\widehat k_{b,h,t}
=\frac{k_{b,h,t}}
{\sqrt{\frac1{d_h}\sum_jk_{b,h,t,j}^2+\epsilon}}.
$$

```python
if self.qk_norm:
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
```

它不增加可训练参数，主要抑制attention logits随训练放大。实验7与baseline1曲线基本一致，因此单独的无参QK-Norm没有明显质量收益。

### 3.2 实验8：QK-Norm后增加learnable per-head scale

QK-Norm会消除Q的整体幅度，因此在norm之后加入每层、每head可学习logit scale（等价于逆温度）：

$$
q'_{l,h}=s_{l,h}\widehat q_{l,h},
\qquad s_{l,h}^{(0)}=1.
$$

最终logits：

$$
A_{l,h}
=\frac{s_{l,h}\widehat Q_{l,h}\widehat K_{l,h}^{\top}}
{\sqrt{d_h}}.
$$

```python
self.qk_scale = nn.Parameter(torch.ones(config.n_head))

qk_scale = self.qk_scale.to(q.dtype).view(1, -1, 1, 1)
q = q * qk_scale
```

新增参数只有：

$$
L\times H=12\times12=144.
$$

实验8相对baseline1改善约0.02，说明重要的不是单纯压平Q/K，而是稳定logits后允许每个head重新学习合适的logit scale。

### 3.3 实验9：head数12→6

保持$C=768$，head数从12改为6：

$$
d_h:64\to128.
$$

QKV和输出矩阵的总参数量不变，attention主要FLOPs的渐近量也不变，只改变kernel形状和head分解。结果与实验8无明显差异，因此继续使用12 heads。

---

## 4. 1-bank Value Embedding（实验10、18、29）

### 4.1 结构定义

新增共享token-indexed value bank：

$$
E^V\in\mathbb R^{V\times C}.
$$

对输入token id $t$查询一次：

$$
e_t^V=E^V[t]\in\mathbb R^C.
$$

只在0-based层8、9、10、11注入attention value：

$$
\widetilde V_{l,h,t}
=V_{l,h,t}+\alpha_{l,h}e^V_{t,h},
\qquad l\in\{8,9,10,11\}.
$$

初始化：

$$
E^V\sim\mathcal N(0,0.02^2),
\qquad \alpha_{l,h}^{(0)}=1.
$$

代码：

```python
# GPT.__init__
transformer_modules['vte'] = nn.Embedding(config.vocab_size, config.n_embd)

# GPT.forward: one lookup shared by selected layers
shared_value_embed = self.transformer.vte(idx)

# Attention.forward
ve = value_embed.to(v.dtype).view(B, T, n_head, head_dim).transpose(1, 2)
gate = self.value_embed_gate.to(v.dtype).view(1, n_head, 1, 1)
v = v + gate * ve
```

### 4.2 参数与计算

默认配置下新增：

$$
VC+4H
=50304\times768+4\times12
=38,633,520
$$

个参数。

虽然参数量大，但运行时只是一次embedding lookup、一次dtype转换和4层逐元素加法，不增加大矩阵乘，因此吞吐影响很小。

### 4.3 消融结论

- 实验10：相对有参QK-Norm进一步改善约0.01；
- 实验18：MoE中移除VE，质量差约0.01，速度无明显提升；
- 实验29：固定时间Dense无VE版本full val `3.041136`，弱于有VE主线。

结论：VE是“大参数、低计算”组件，保留。

---

## 5. Residual projection tiny initialization（实验11）

实验11把attention和MLP的所有`c_proj`初始化标准差再乘：

$$
\texttt{resid\_init\_scale}=0.1.
$$

因此：

$$
\sigma_{\text{tiny}}
=0.1\times\frac{0.02}{\sqrt{2L}}
\approx4.0825\times10^{-4}.
$$

```python
if pn.endswith('c_proj.weight'):
    nn.init.normal_(
        p,
        mean=0.0,
        std=0.02 / math.sqrt(2 * config.n_layer) * config.resid_init_scale,
    )
```

它不改变参数量或运行图，只让每个residual branch在训练初期更接近关闭状态：

$$
X_{l+1}\approx X_l.
$$

实验11改善约0.01，随后所有主线模型均保留`resid_init_scale=0.1`。

---

## 6. Dense FFN：GELU → ReLU² → SwiGLU（实验12、19、22、30）

### 6.1 实验12：ReLU²

保持FFN矩阵形状`768→3072→768`不变，只替换激活：

$$
\phi(x)=\operatorname{ReLU}(x)^2.
$$

```python
x = self.c_fc(x)
x = F.relu(x).square()
x = self.c_proj(x)
```

早期代码使用：

```python
F.relu(x) ** 2
```

Speedrun代码改为`.square()`，避免`pow()`在autocast下提升到FP32并增加activation traffic。

实验12改善约0.04，是已测试Dense结构中最大的单项收益。

### 6.2 等参数Dense SwiGLU

SwiGLU形式：

$$
\operatorname{SwiGLU}(x)
=W_2\left[\operatorname{SiLU}(W_gx)\odot(W_ux)\right].
$$

普通FFN有2个大矩阵；SwiGLU有3个。为保持参数与主矩阵计算近似相同：

$$
2C\cdot3072=3C\cdot D_{\text{SwiGLU}},
$$

得到：

$$
D_{\text{SwiGLU}}=2048.
$$

对应代码形态：

```python
gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
x = F.silu(gate) * up
x = self.down_proj(x)
```

实验30的固定时间结果为`3.040597`，没有胜过ReLU²主线；实验31通过accum8改善到`3.034069`，但仍不能证明SwiGLU结构优于ReLU²，因为batch/update频率同时改变。

结论：固定A100时间预算下保留ReLU²。

---

## 7. Gated Attention与fused QKVG（实验13及后续工程化）

QKV投影实现经历了以下工程演进；这几步需要和质量变量区分：

1. baseline1沿用nanoGPT单个`[C,3C]` fused QKV投影；
2. baseline2为让Muon分别正交化Q/K/V，暂时拆成独立`wq/wk/wv`；
3. baseline3重新合成`[C,3C]` QKV，同时以独立`[C,C]`投影加入attention gate；
4. baseline4把Q/K/V/gate进一步拼成`[C,4C]` QKVG，并由Muon按4个逻辑row blocks处理。

### 7.1 实验13：SDPA输出后elementwise sigmoid gate

最初实现中，QKV和gate是两个投影：

```python
self.wqkv = nn.Linear(C, 3 * C, bias=False)
self.gate = nn.Linear(C, C, bias=False)
```

计算：

$$
Y=\operatorname{SDPA}(Q,K,V),
$$

$$
G=\sigma(XW_g),
$$

$$
Y'=Y\odot G.
$$

```python
y = scaled_dot_product_attention(q, k, v, is_causal=True)
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = y * torch.sigmoid(self.gate(x))
y = self.c_proj(y)
```

这是逐token、逐channel的query-dependent gate；不同head对应不同通道块。新增参数：

$$
LC^2=12\times768^2=7,077,888.
$$

实验13进一步改善约0.01，成为baseline3。

### 7.2 Speedrun工程化：融合为一个QKVG GEMM

后续将Q、K、V和gate投影拼成：

$$
W_{QKVG}\in\mathbb R^{4C\times C}.
$$

```python
self.wqkv = nn.Linear(C, 4 * C, bias=False)
q, k, v, g = self.wqkv(x).split(C, dim=-1)
...
y = y * torch.sigmoid(g)
```

数学与“独立QKV + gate Linear”一致，但从两个input GEMM减少到一个。Muon中仍按4个逻辑row blocks分别正交化：

```python
{'params': qkvg_params, 'muon_split_rows': 4}
```

这是工程优化而不是单独质量实验；从baseline4开始均采用fused QKVG。

---

## 8. MoE主线（实验14–19、21–24）

### 8.1 标准4专家top-1 MoE

每个Dense FFN被4个expert替换：

$$
f_e(x)=W_{2,e}\phi(W_{1,e}x).
$$

Router读取block的pre-normalized hidden：

$$
p(x)=\operatorname{softmax}(W_rx),
\qquad e^*=\arg\max_ep_e(x).
$$

Top-1输出仍乘选中专家的softmax概率：

$$
\operatorname{MoE}(x)=p_{e^*}(x)f_{e^*}(x).
$$

注意top-1权重没有重新归一化为1，否则主任务对router概率的梯度会消失。

实现：

```python
probs = F.softmax(self.router(xf), dim=-1)
top_p, top_i = probs.topk(1, dim=-1)

flat_expert = top_i.reshape(-1).to(torch.int32)
perm = torch.argsort(flat_expert)
sorted_x = xf[flat_token[perm]]
counts = torch.bincount(flat_expert, minlength=n_experts)
offs = torch.cumsum(counts, dim=0).to(torch.int32)

h = torch._grouped_mm(sorted_x, w1, offs=offs)
h = F.relu(h).square()  # or GELU
y = torch._grouped_mm(h, w2, offs=offs)
out[flat_token[perm]] = y * top_p.reshape(-1, 1)[perm]
```

所有token均被分配，没有capacity限制或token drop。

### 8.2 Switch-style load balancing loss

定义：

$$
f_e=\frac{\#\{\text{assignments to }e\}}{N},
\qquad P_e=\frac1N\sum_np_{n,e}.
$$

辅助损失只在training且开启梯度时计算；validation/full validation不加入该项：

$$
L_{\text{aux}}
=E\sum_ef_eP_e.
$$

训练目标：

$$
L=L_{\text{CE}}+0.01\sum_lL_{\text{aux},l}.
$$

```python
f = counts.to(probs.dtype) / N
P = probs.mean(dim=0)
aux = n_experts * (f * P).sum()
```

### 8.3 Expert width消融

对于4专家、top-1、hidden $C=768$，expert总参数和active主MAC如下：

| Expert dim $D$ | 每层expert总参数 | 每token active MAC代理 | 对应实验 |
| -------------: | ---------------: | ---------------------: | -------- |
|           3072 |          18.874M |                 4.719M | 14       |
|           2048 |          12.583M |                 3.146M | 15、16   |
|           1536 |           9.437M |                 2.359M | 17、21   |

计算公式：

$$
P_{\text{experts}}=4\times2CD,
\qquad \operatorname{MAC}_{\text{active}}=2CD.
$$

Router另有$C\times4$参数，量级可忽略。

初始化：

```python
nn.init.normal_(w1, std=0.02)
nn.init.normal_(w2, std=0.02 / sqrt(2 * n_layer) * resid_init_scale)
```

### 8.4 MoE激活和SwiGLU

实验15使用ReLU²，实验16使用GELU，二者无明显区别。

实验19/22使用等参数MoE SwiGLU。对于ReLU² expert dim $D_r$，等参数SwiGLU width满足：

$$
2CD_r=3CD_s
\quad\Rightarrow\quad
D_s\approx\frac23D_r.
$$

因此：

- 相对`D_r=2048`，数学等参数宽度为`1365.3`；早期笔记指向为硬件对齐取约`1368`，但当前`report/`目录缺少实验19原始模型文件，不能逐源码确认；
- 相对`D_r=1536`，后续Speedrun归档实现可确认使用`D_s=1024`。

SwiGLU可小幅改善MoE样本效率，但引入第三个expert矩阵和额外逐元素门控，固定时间下没有追上Dense。

### 8.5 MoE最终结论

理论active MAC下降并没有转化为A100吞吐提升，原因包括：

- router softmax与top-k；
- assignment sort / bincount / prefix offsets；
- 小型、不均匀expert grouped GEMM；
- scatter combine；
- 更多参数的optimizer更新。

实测Dense MFU约50%，MoE约34%。因此在4小时墙钟目标下停止MoE主线。

---

## 9. Efficient LatentMoE（实验26）

### 9.1 结构

将原hidden payload压缩到latent space，但router仍读取完整hidden：

$$
p(x)=\operatorname{softmax}(W_rx),
\qquad W_r:\mathbb R^{768}\to\mathbb R^8.
$$

$$
z=W_{\downarrow}x,
\qquad W_{\downarrow}:768\to384.
$$

选择8个latent experts中的top-1：

$$
y_z=p_{e^*}(x)\,W_{2,e^*}\phi(W_{1,e^*}z),
$$

其中：

$$
W_{1,e}:384\to1536,
\qquad W_{2,e}:1536\to384.
$$

最终共享up projection：

$$
y=W_{\uparrow}y_z,
\qquad W_{\uparrow}:384\to768.
$$

代码：

```python
probs = F.softmax(self.router(xf), dim=-1)  # router sees 768-d x
top_p, top_i = probs.topk(1, dim=-1)
zf = self.down_proj(xf)                     # payload becomes 384-d
...
y = grouped_mm(grouped_mm(sorted_z, w1), w2)
out = self.up_proj(out_latent)
```

Router不读取latent representation，是为了避免压缩瓶颈损失路由判别信息。

### 9.2 预算

配置：

```text
latent_dim = 384
num_experts = 8
top_k = 1
expert_dim = 1536
```

Latent expert参数：

$$
8\times2\times384\times1536=9.437M,
$$

与标准4专家、D=1536完全相同。

共享down/up新增：

$$
2\times768\times384=0.590M.
$$

每token active主矩阵MAC（不含router的`768×8=6,144` MAC及sort/dispatch/scatter开销）：

$$
2\times768\times384+2\times384\times1536
=1,769,472\approx1.769M,
$$

约为标准MoE主矩阵MAC的75%。

初始化：

```python
down_proj: std = 0.02
expert w1/w2: std = 0.02
up_proj: std = 0.02 / sqrt(2L) * resid_init_scale
```

只有最终up projection使用tiny residual init，避免expert `w2`和up同时缩小。

实验26中途曲线明显变差并停止；单A100又没有all-to-all通信收益，因此淘汰。

---

## 10. Residual结构实验：readout backout与embedding shortcut（实验24、25、27）

### 10.1 Final readout backout

缓存block index 7执行后的hidden $X_8$，最终与$X_{12}$混合：

$$
Z=a\operatorname{LN}_f(X_{12})+b\operatorname{LN}_f(X_8).
$$

使用同一个`ln_f`参数归一化两路，不新增第二个LayerNorm。

初始化：

$$
[a,b]=[1,0],
$$

因此初始输出与原baseline严格相同。

```python
if layer_idx == 7:
    backout_hidden = x

final_norm = self.transformer.ln_f(x)
skip_norm = self.transformer.ln_f(backout_hidden)
x = mix[0] * final_norm + mix[1] * skip_norm
```

新增参数只有2个；运行时多一次final LayerNorm、两次标量乘和一次加法。

实验24在MoE上观察到小幅收益；实验27在Dense单次run中从`3.036966`到`3.035213`，点估计改善约0.00175，但没有多seed重复，不能断言该千分位差异具有统计显著性。Dense residual stream本来已隐式保留早期hidden，因此该分支高度冗余；当前可保留，但不再重点投入。

### 10.2 Identity-init embedding shortcut

记初始token+position embedding经过dropout后的状态为$X_0$。每个block入口先执行：

$$
\widetilde X_l=a_lX_l+b_lX_0,
\qquad[a_l,b_l]_{\text{init}}=[1,0].
$$

然后：

$$
X_{l+1}=\operatorname{Block}_l(\widetilde X_l).
$$

```python
self.embedding_shortcut_mix = nn.Parameter(torch.tensor([1.0, 0.0]))

mix = self.embedding_shortcut_mix.to(x.dtype)
x = mix[0] * x + mix[1] * x0
```

12层新增24个标量，但需全程保存$X_0$，且每层增加大张量mul/add。实验25结果`3.056808`，固定时间无优势，淘汰。

---

## 11. VE、Embedding decay与参数删除消融（实验18、28、29）

### 11.1 Embedding weight decay

这不是结构改变，但影响embedding类参数的训练：

- `wte/wpe`使用独立AdamW group；
- `vte`使用独立group，支持`value_embed_decay`与`value_embed_lr_scale`；
- `wte`与LM head权重绑定，因此改变wte decay也同时改变输出分类器的正则化。

实验28单独关闭embedding decay没有改善：`3.036915` vs baseline4 `3.035213`。后期当前best虽使用decay 0，但它同时改变了accumulation、LR和momentum，不能把收益单独归因给embedding no-decay。

### 11.2 完全删除VE

无VE版本删除：

- `transformer.vte.weight`；
- 最后4层的`value_embed_gate`；
- forward中的lookup、dtype cast和value注入。

移除参数：

$$
38,633,520.
$$

但主要节省显存和optimizer state；主干Attention、FFN和大词表LM head计算不变，因此速度差异小。实验29full val变差到`3.041136`，保留VE。

---

## 12. Affine RMSNorm消融（实验43）

将每层两个pre-LayerNorm及最终`ln_f`替换为Affine RMSNorm：

LayerNorm：

$$
\operatorname{LN}(x)
=\gamma\odot\frac{x-\mu(x)}{\sqrt{\operatorname{Var}(x)+\epsilon}}+\beta.
$$

RMSNorm：

$$
\operatorname{RMSNorm}(x)
=\gamma\odot\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}+\beta.
$$

正式scratch配置`bias=False`，因此只有可学习gain $\gamma$，没有shift $\beta$。

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, eps=1e-5)
```

替换norm数量：

$$
2L+1=25.
$$

匹配结果：

| Norm           | Full val | Tokens |
| -------------- | -------: | -----: |
| LayerNorm      | 3.013418 | 2.506B |
| Affine RMSNorm | 3.013878 | 2.509B |

单次匹配run中RMSNorm快约0.14%，full val高0.00046；由于没有多seed重复，这一差异统计上未定，只能描述为点估计平局偏负。最终基于更简单的证据链保留LayerNorm。

---

## 13. Speedrun评估框架（实验20之后，非模型结构）

虽然Speedrun不是前向结构，但它决定结构比较是否公平。

### 13.1 墙钟停止

训练时间是唯一停止条件：

```python
while speedrun_mode or iter_num < target_num_updates:
    ...
    if speedrun_elapsed >= speedrun_time_limit_seconds:
        break
```

`max_iters`在Speedrun中只作为LR schedule horizon，不会提前终止更快模型。

### 13.2 Warmup后估计horizon

用warmup后半段的同步测速：

$$
\bar t
=\frac{t_{\text{warmup end}}-t_{\text{probe start}}}
{N_{\text{measured updates}}}.
$$

剩余update估计：

$$
N_{\text{remain}}
=\left\lceil\frac{T_{\text{limit}}-T_{\text{elapsed}}}{\bar t}\right\rceil.
$$

Nominal horizon再增加约1% headroom：

$$
N_{\text{schedule}}
=N_{\text{nominal}}+
\max\left(2,\left\lceil0.01N_{\text{nominal}}\right\rceil\right).
$$

### 13.3 Cosine与WSD

Cosine：

$$
\eta(t)=\eta_{\min}
+\frac12\left[1+\cos(\pi r)\right]
(\eta_{\max}-\eta_{\min}).
$$

WSD在warmup后先保持peak LR，在最后比例$f$进入cooldown：

$$
t_{\text{decay}}=N_{\text{schedule}}(1-f).
$$

Linear cooldown：

$$
\eta=\eta_{\min}+(1-r)(\eta_{\max}-\eta_{\min}).
$$

Cosine cooldown：

$$
\eta=\eta_{\min}
+\frac12(1+\cos\pi r)(\eta_{\max}-\eta_{\min}).
$$

### 13.4 WSD最终实证结果

在相同baseline4结构、单卡4小时、batch 24、accum8和Muon momentum 0.90下，WSD linear 30% cooldown形成当前最优结果：

| 实验 | Peak AdamW / Muon LR | Schedule       | Embedding decay |     Full val | Tokens |
| ---: | -------------------- | -------------- | --------------: | -----------: | -----: |
|   37 | `.00234 / .026`      | cosine         |               0 |     3.012101 | 2.521B |
|   46 | `.00234 / .026`      | WSD linear .30 |       W&B未记录 |     3.008401 | 2.511B |
|   47 | `.003 / .03`         | WSD linear .30 |               0 | **3.005883** | 2.496B |
|   48 | `.0036 / .04`        | WSD linear .30 |             0.1 |     3.017362 | 2.496B |

WSD的收益不是简单来自处理更多tokens：实验47的tokens略少于实验37，但full val降低约`0.006218`。在已完成候选中，`.003/.03`组合结果最好；不过实验46缺少embedding-decay记录，实验48又同时使用了`embedding_decay=.1`，因此46–48不能被解释为严格的单变量peak-LR扫描。能够直接确认的是：实验47这套完整组合当前最优。

实验47结束时schedule progress约为`0.9903`，AdamW LR约为`1.016e-4`；由于动态horizon保留headroom，墙钟截止时不会精确到达`min_lr=1e-6`，但不同候选使用同一估计规则，比较口径一致。

最终full validation位于计时区外，checkpoint先原子保存，再写远程summary。

---

## 14. 结构迭代总表

|   实验 | 结构改动                          |                     参数变化 | 运行代价            | 结论                 |
| -----: | --------------------------------- | ---------------------------: | ------------------- | -------------------- |
|      7 | Parameter-free QK-Norm            |                            0 | 小                  | 中性                 |
|      8 | QK-Norm + per-head scale          |                         +144 | 小                  | 明确正收益           |
|      9 | 12 heads→6 heads                  |                            0 | kernel形状变化      | 无明显收益           |
|     10 | 1-bank VE                         |                     +38.634M | lookup + add        | 小幅正收益           |
|     11 | residual projection tiny init     |                            0 | 0                   | 正收益               |
|     12 | GELU→ReLU²                        |                            0 | 近似等价/略快       | 强正收益             |
|     13 | SDPA output gate                  |                      +7.078M | 1个额外投影，后融合 | 小幅正收益           |
|  14–19 | 4-expert top-1 MoE及激活/宽度消融 |                     大幅增加 | A100利用率下降      | 固定时间失败         |
| 24、27 | Final readout backout             |                           +2 | 额外LN与mix         | MoE小幅正；Dense边际 |
|     25 | Embedding shortcut                |                          +24 | 每层大张量mix       | 固定时间负           |
|     26 | Efficient LatentMoE               | 每层约+0.59M vs标准D1536 MoE | 理论MAC低、实际复杂 | 曲线差，停止         |
|     29 | 删除VE                            |                     −38.634M | 主干计算不变        | 质量变差             |
|     30 | Dense等参数SwiGLU                 |                 参数近似不变 | 第三矩阵/门控       | 固定时间负           |
|     43 | Affine RMSNorm                    |                        近似0 | 略快                | 与LN打平偏负         |

---

## 15. 当前最终结构

当前正式推荐结构仍为baseline4的LayerNorm Dense模型：

```python
# Embedding
x = dropout(wte(idx) + wpe(pos))
ve = vte(idx)  # one shared bank

for layer_idx, block in enumerate(blocks):
    # pre-LN attention
    u = block.ln_1(x)
    q, k, v, g = block.attn.wqkv(u).split(C, dim=-1)

    q = rms_norm_per_head(q)
    k = rms_norm_per_head(k)
    q = q * block.attn.qk_scale

    if layer_idx in (8, 9, 10, 11):
        v = v + block.attn.value_embed_gate * ve

    y = scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y * sigmoid(g)
    x = x + block.attn.c_proj(y)

    # pre-LN ReLU² FFN
    h = block.mlp.c_fc(block.ln_2(x))
    h = relu(h).square()
    x = x + block.mlp.c_proj(h)

    if layer_idx == 7:
        x_block7 = x

# identity-init final readout
x = readout_mix[0] * ln_f(x) + readout_mix[1] * ln_f(x_block7)
logits = lm_head(x)  # tied with wte
```

关键初始化：

```text
ordinary weights:                std = 0.02
attention/MLP c_proj:            std = 0.02 / sqrt(24) × 0.1
qk_scale:                        1.0
VE gate:                         1.0
readout mix:                     [1.0, 0.0]
LayerNorm gain:                  1.0
bias:                            disabled
```

当前约170.09M参数分解（含position embedding）：

```text
tied token embedding / LM head:  38.633M
position embedding:               0.786M
12 Transformer blocks:           92.031M
1-bank VE + four VE gates:        38.634M
final LayerNorm + readout mix:     0.001M
------------------------------------------------
total:                           170.085M
```

当前该结构的最佳已完成Speedrun为实验47（`ixd9wd56`）：

```text
batch_size = 24
gradient_accumulation_steps = 8
AdamW LR = 0.003
Muon LR = 0.03
Muon momentum = 0.90
Muon weight decay = 0.01
embedding_decay = 0.0
value_embed_decay = 0.0
warmup_iters = 100
min_lr = 1e-6
lr_schedule = wsd
wsd_decay_frac = 0.30
wsd_decay_style = linear
speedrun_time_limit_seconds = 14400
val/final_loss_full = 3.005883
```

这更新了此前实验37的cosine推荐；模型结构未改变，变化仅来自优化器峰值LR与调度策略。

---

## 16. 不确定性与待补档项

1. `report/`目录没有保存实验14–19每一版早期MoE源文件；本文MoE公式依据后续保持同语义的Speedrun实现和 `experiment.md` 重建。若需要对实验14–19逐行复现，建议补充当时原始model文件。
2. 实验19早期MoE SwiGLU的数学等参数width为$4096/3\approx1365.3$。早期笔记指向可能对齐到1368，但当前归档缺少该实验原始源码，故本文不把1368视为已逐源码确认的事实。
3. Dense SwiGLU实验30的当前归档目录未保留对应模型文件；本文按等参数条件确定hidden width为2048，这与此前运行命令/设计一致。

如果能补充实验14–19和实验30的原始模型文件，可以进一步把上述三项从“重建”升级为逐行、逐tensor完全核对。
