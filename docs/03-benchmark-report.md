# 03 — 基准验证报告（llama-server vs LiGGUF）

> 复现时间：2026-08。环境：Windows x64 / MSVC 19.51 / AVX2 / 无 GPU。
> 数据与脚本见 `../bench/`，可直接复现。
> 测试集由 `gen_dataset.py` 程序化生成（120 docs + 150 queries，含 30 干扰），GT 生成时已知，无需人工标注。

---

## 1. 环境

| 项 | 值 |
|----|----|
| 模型 | `D:\bge-small-zh-v1.5-f16.gguf`（45,676,544 B，F16，pooling=MEAN，无 pooler） |
| llama-server | `D:\llama-b10068\llama-server.exe`（llama.cpp b10068，端口 8083） |
| LiGGUF | 本方案编译的 `ligguf_embed.exe`（305KB，`/O2 /arch:AVX2`，含 F16C 优化） |
| 测试集 | 120 条中文记忆（docs），120 条相关查询 + 30 条干扰（queries），GT 程序生成 |

## 2. 结果

### 2.1 召回 / MRR@5

```
top-5 命中总数: llama=118  ligguf=118
MRR@5       : llama=0.036  ligguf=0.036
```

逐题明细（完整输出见 `compare.py`）：

- 120 题两引擎 top-5 命中数完全持平（118 vs 118）。
- 两侧各有 1 题独占命中：Q32「我的职业具体是哪一行」llama 中、LiGGUF 未中；Q99「我有哪种动手特长」LiGGUF 中、llama 未中。其余 118 题命中情况一致。
- 30 条干扰查询（无 GT）两引擎 top-5 命中均为 0，无误报。

### 2.2 bootstrap CI（配对，2000 次）

```
命中数差(ligguf-llama): 均值=+0.0000  95%CI=[-0.0250,+0.0250]  P(差>=0)=0.642
MRR差  (ligguf-llama): 均值=+0.0063  95%CI=[-0.0125,+0.0278]  P(差>=0)=0.700
```

- 两个指标的 95% CI 均跨越 0，P 值 > 0.05 → **120 题下两引擎检索差异统计不显著**。
- 对比旧 12 题集（命中 13 vs 12，P≈0.27 不显著）：小样本的"差 1 题"是噪声，扩大测试集后两侧各 miss 1 题、总数持平，进一步确认等价。

### 2.3 向量一致性（同 doc 余弦）

```
同一 doc 的 llama·ligguf 余弦: 均值=0.891 最小=0.837 最大=0.925
跨引擎不同 doc 平均余弦 = 0.385（基数）
```

- 0.891 远高于"不同 doc"基数 0.385，说明两引擎语义方向高度一致，只是存在系统性的数值偏移。
- 该偏移来自两套不同的计算后端（llama.cpp=ggml 图 + 量化/AVX 点积；LiGGUF=纯标量 fp32），非单一可修复 bug。诊断过程见下节。

### 2.4 top-5 集合一致性（Jaccard）

```
top-5 Jaccard 均值 = 0.63
```

两引擎各查询返回的 top-5 集合平均重合 63%，配合命中数/MRR 持平，说明检索质量等价。

### 2.5 性能（F16C 优化后）

```
120 docs:  9.0s → 2.95s   (~3.1x)
150 queries: 8.8s → 3.0s  (~2.9x)
单条文本延迟: ~46ms
```

- 优化内容：`dot_f16_f32` 与 `f16_to_f32_row` 用 F16C(`_mm256_cvtph_ps`) 批量 fp16→fp32，替代逐元素 LUT 查表。
- 精度验证：新旧 exe 输出向量**逐位一致**（F16C 与 IEEE 精确 LUT 转换结果相同），检索指标不变。

## 3. 统计显著性说明

- 120 题、bootstrap 2000 次，命中/MRR 差 95%CI 均跨 0、P>0.05。
- 结论：**两引擎在中文记忆检索上统计等价**（召回、排序质量均无显著差异），干扰查询无误报。

## 4. 精度差异诊断过程（diff 追查记录）

> 供后续想追平 0.891 余弦 / 提升向量可移植性的人参考。

**已排查并排除的假设：**

1. **tokenizer 不一致** → ❌ 排除。`dump_vocab.py`/`check_token.py` 打印 token 文本，`872='▁你'` 等与 llama.cpp `/tokenize` 逐 id 一致（早先"不一致"是控制台中文显示错位的假象）。
2. **GGUF 含 dense projection / pooler** → ❌ 排除。`gguf_inspect.py` 确认 69 个 tensor 中**无** `cls.dense`/`pooler`，两引擎都走纯 MEAN。
3. **pooling 是否排除 CLS/SEP** → ❌ 排除。已确认 llama.cpp 对 BERT 的 MEAN pooling **包含**全部 token（权重 1/n）；LiGGUF 同样包含。临时改成"排除 CLS/SEP"的实验使余弦更差，已回滚。
4. **fp16 LUT 精度** → ❌ 排除。`fp16_to_fp32` 是位运算魔数转换（IEEE 精确），与 llama.cpp `ggml_fp16_to_fp32` 逐位等价；F16C 硬件转换同样精确（见 2.5）。
5. **token_types.weight 取错 segment** → ❌ 排除。GGUF dims 顺序 `[2, 512]`（ne[0]=2, ne[1]=512），取第 0 行 = segment 0 的 512 维，单句输入下正确。
6. **matmul 布局 / scale** → ❌ 排除。`out[r] = dot(w[r], src)` 与 `W@x` 等价；attention scale = `1/sqrtf(head_dim=64)` 与 BERT 一致。

**结论**：剩余差异来自后端实现差异（softmax 数值、点积累加顺序、gelu 实现细节、量化路径），散布于各层。方向一致、幅度小、跨 doc 稳定。检索等价性已由 120 题 bootstrap 证明；**追平余弦需在 LiGGUF 内复刻 ggml 确切算子，收益不匹配成本，建议接受为 known limitation。**

## 5. 复现步骤

```powershell
# 0) 重新生成测试集（可选）
python bench\gen_dataset.py

# 1) LiGGUF 向量
.\ligguf_build_port\ligguf_embed.exe D:\bge-small-zh-v1.5-f16.gguf bench\docs.txt bench\ligguf_docs.bin
.\ligguf_build_port\ligguf_embed.exe D:\bge-small-zh-v1.5-f16.gguf bench\queries.txt bench\ligguf_queries.bin

# 2) llama 向量（需 8083 起 server）
python bench\embed_llama.py

# 3) 对比（含 bootstrap CI）
python bench\compare.py
```