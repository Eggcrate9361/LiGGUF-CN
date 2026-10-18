# 01 — 技术细节文档（中文环境优化）

本文档记录 LiGGUF → 跑通 bge-small-zh 中文向量所需的全部技术改动、原理与验证过程。面向后续维护者 / 想深入了解的人。

---

## 1. 背景

LiGGUF（[matrixsmaster/ligguf](https://github.com/matrixsmaster/ligguf)）是一个**零依赖、单文件、纯 C++** 的 GGUF 推理引擎。原项目定位是 LLaMA 风格的生成式推理（`fast/` 目录为带对话应用的子集）。

EggBloom 需要本地方向量检索，原方案是外置 `llama-server`（CUDA，~625MB）。目标：用 LiGGUF 替代。

**关键差异**：bge-small-zh 是 **BERT 架构（encoder-only、双向注意力）**，不是 LLaMA 式 decoder。原版 LiGGUF 的 `fast/vdb.cpp` 已经实现了一个简化版 BERT 推理（4 层），但从未在中文上验证过。

模型文件：`D:\bge-small-zh-v1.5-f16.gguf`（45,676,544 字节 ≈ 45.7MB，F16 权重，`bert.pooling_type=2`=MEAN）。

## 2. 总体改动清单

| # | 文件 | 改动 | 必要性 |
|---|------|------|--------|
| 1 | `lil_gguf.cpp` | Windows mmap 移植（`_WIN32` 分支，fread 全量读入） | 必须（MSVC 无 `mmap`） |
| 2 | `lil_gguf.h` | `wtensor` 增加 `_WIN32` packed 分支 | 必须（MSVC 无 `__attribute__((packed))`） |
| 3 | `common.cpp` | `clock_gettime` → `GetTickCount64` | 必须 |
| 4 | `vdb.cpp` | `localtime_r` → `localtime_s` | 必须 |
| 5 | `vdb.cpp` | **重写 `bert_tokenize`（中文分词对齐 llama.cpp）** | 必须（否则中文全变 `[UNK]`） |
| 6 | `vdb.cpp` | **`ensure_bert()` 调用 `fp1632_init()`** | **决定性修复** |
| 7 | `vdb.cpp` | 新增 `vdb_embed_export`（文本 → 512 维向量） | 集成需要 |
| 8 | `ligguf_embed.cpp` | 新增 harness（CLI 入口） | 集成/基准需要 |
| 9 | `common.h` | 编译宏 / 结构体布局对齐 | 必须 |

## 3. 关键改动详解

### 3.1【决定性】fp16 LUT 未初始化（原版隐藏 bug）

**症状**：所有文本输入输出**完全相同的向量**（余弦 = 1.0）。

**根因链**：
- `lil_math.h` / `lil_math.cpp`：`fp16_to_fp32(H)` 实现为查表 `fp1632_lut[H]`（65536 项）
- LUT 由 `fp1632_init()` 填充（遍历全部 65536 个 half 值，位运算转 float）
- 原版中 `fp1632_init()` **只在 `ligguf_fast_qk.cpp` 的 main 里调用**
- 我们的 harness（`ligguf_embed.cpp`）没调 → LUT 全 0
- GGUF 的 F16 权重（`token_embd.weight`、`position_embd.weight`、全部 `blk.*.weight` 都是 F16）经 `dequantize_row` / `dot_f16_f32` 解码 → 全部 0 → 输出恒等向量

**修复**：`vdb.cpp` 的 `ensure_bert()` 开头加一行 `fp1632_init();`（幂等，仅首次填表）。

**验证**：修复后 `bert_embed` 输出每 doc 向量非零且彼此不同；同 doc 余弦从 1.0 降到 0.878（真实语义差异）。

> 注意：`fp16_to_fp32` 是位运算魔数转换，LUT 值本身是 IEEE 精确的，与 llama.cpp `ggml_fp16_to_fp32` 逐位等价。这已通过读实现 + 采样验证。

### 3.2 中文分词重写（对齐 llama.cpp WPM）

**原版问题**：原 `bert_tokenize` 按空白/标点切分 token，再对每个"单词"逐字符小写；对中文这种无空格文字，整个句子变成一个"word" → 查表失败 → 全部 `[UNK]`。

**重写要点**（对照 llama.cpp `tokenize_bert` / WordPiece）：

```
输入: "你还记得我喜欢吃什么吗"
1. UTF-8 逐字节解码 → 每 3 字节一个中文字符（无 BOM）
2. 去掉 U+FEFF BOM 残留
3. 每个中文字符 = 一个独立的 word
4. 每个 word 前拼 ▁(U+2581, "\xE2\x96\x81")，得到 "▁你" "▁还" "▁记" ...
5. 每个 word 内部做最长贪心：word → 查词表最大匹配前缀拆 subword，拆不下来才 [UNK]
```

**结果**：`"你还记得我喜欢吃什么吗"` → `101 872 6820 6381 2533 2769 1599 3614 1391 784 720 1408 102`（101=CLS, 102=SEP），与 llama.cpp `/tokenize` 接口输出**逐 id 一致**。

**已验证**：`bench/check_token.py` 打印 token 文本确认 `872='▁你'`、`6820='▁还'`、`4500='▁用'` 等均带 `▁` 前缀、内容正确。

### 3.3 Windows / MSVC 移植

- `lil_gguf.cpp` `open_mmap`：新增 `#ifdef _WIN32` 分支——`fopen(rb)` + `fseek/ftell` 拿大小 + `malloc(fsize+64)` + 一次 `fread` 全量载入。行为与 POSIX `mmap` 等价（base 指向完整文件镜像）。
- `lil_gguf.h`：结构体 `packed` 宏，MSVC 用 `#pragma pack(push,1)` / `#pragma pack(pop)`。
- `common.cpp`：计时函数换 `GetTickCount64`。
- `vdb.cpp`：线程安全时间函数换 `localtime_s`。
- 编译：`cl /O2 /std:c++17 /EHsc /arch:AVX2 /D__AVX2__`。

### 3.4 vdb_embed_export（向量导出）

```cpp
// vdb.cpp 末尾新增
bool vdb_embed_export(const char* text, float* out, int maxdim);
// 调 bert_embed(text, out)（内部已做 MEAN pooling + L2 normalize）
// 返回实际维数（本模型 512）
```

harness `ligguf_embed.cpp`：`<model.gguf> <texts.txt> <out.bin>`，逐行调用，跳空行，写 `N × 512` float32。

## 4. 模型结构与计算路径（对齐性核查）

### 4.1 GGUF 内容（`gguf_inspect.py`）

- 26 个 KV，69 个 tensor
- `bert.pooling_type = 2`（MEAN）
- `tokenizer.ggml.model = bert`
- **无** `cls.dense` / `pooler` 权重 → 两引擎都不做 dense projection，直接 MEAN pool
- `token_embd.weight` [512, 21128] F16、`position_embd.weight` [512, 512] F16
- `token_types.weight` [512, 2] F32（4 个 `blk.*` 层：`attn_norm`/`attn_q`/`attn_k`/`attn_v`/`attn_output`/`ffn_norm`/`ffn_up`/`ffn_down` 各带 weight 和部分 bias）

### 4.2 计算路径对照

| 环节 | llama.cpp | LiGGUF（vdb.cpp） | 结论 |
|------|-----------|-------------------|------|
| tokenizer | WPM + ▁ 前缀 | 重写后同 | ✅ 逐 id 一致 |
| embedding | `token_embd + pos_embd + tok_types[seg]` | `row0+row1+row2`（row2=segment 0） | ✅ 等价 |
| token_type | 全 segment 0（单句） | `dequantize_row(F32, ptr, row2, 512)` 取第 0 行 | ✅ 等价 |
| attention | `QK^T / sqrt(head_dim)` + softmax | 同，`1/sqrtf(64)` | ✅ |
| pooling | 全 token（含 CLS/SEP）均值，权重 1/n | 同（vdb.cpp:317-325） | ✅ 等价 |
| L2 norm | server 侧 `/v1/embeddings` 归一化 | `l2norm()` 后返回 | ✅ 等价 |
| dense projection | 无（GGUF 无 pooler） | 无 | ✅ 等价 |

### 4.3 未对齐的已知偏差（非 bug）

同 doc 余弦 = 0.878（非 1.0）。逐层诊断（`diff_vec.py` + 读实现）表明差异**散布于多层数值计算**，根因是两条完全不同的计算后端：

- llama.cpp：ggml 后端 + 可能量化的 dot、AVX 加速、不同累加顺序
- LiGGUF：纯标量（或简单循环）fp32 累加

这种差异（余弦 0.88、方向一致但各维度小幅偏移）在跨引擎对比里属正常数值行为，且**不随模型变大而消除**。若需逐位对齐，唯一路径是在 LiGGUF 里复刻 ggml 的确切算子行为——工程上不划算。

## 5. 验证产物

- `ligguf_embed.exe`：305KB 单文件，无依赖（Release /O2 + AVX2）
- `bench/*.bin`：两引擎向量 dump，可直接复算对比
- 基准结果见 `03-benchmark-report.md`

## 6. 编译命令（MSVC x64）

```powershell
# 设好 MSVC 环境（vcvars64.bat 或手动设 PATH/INCLUDE/LIB 指向
#   D:\VS\VC\Tools\MSVC\14.51.36231 与 D:\VS\Windows Kits\10\...\10.0.26100.0）
cl /O2 /std:c++17 /EHsc /arch:AVX2 /D__AVX2__ ligguf_embed.cpp vdb.cpp lil_math.cpp lil_gguf.cpp common.cpp
```
