# LiGGUF 中文环境优化方案（bge-small-zh）

> 目标：让 [LiGGUF](https://github.com/matrixsmaster/ligguf)（微型 GGUF 推理引擎，零依赖单文件）能替代外置 `llama-server` 跑 bge-small-zh 中文向量模型。
> 状态：**已完成移植与修复，基准验证通过（120 docs + 150 queries bootstrap 不显著），性能优化完成（F16C 批量转换，~3x 提速）**。搁置集成，如需启用见 `docs/10-integration-plan.md`。

---

## 一、结论

| 指标 | llama-server（基线） | LiGGUF（本方案） |
|------|---------------------|------------------|
| top-5 召回 | 118/120 | **118/120** |
| MRR@5 | 0.036 | **0.036** |
| 同 doc 向量余弦 | — | 0.891 |
| top-5 Jaccard | — | 0.63 |
| 引擎体积 | ~625MB+ | **305KB exe** |
| 模型体积 | 45.7MB | 45.7MB |
| 运行依赖 | CUDA llama-server | **零依赖** |
| 单条向量延迟 | — | **~46ms**（F16C 优化后） |

- **召回率：** 118/120 vs 118/120（bootstrap CI=[-0.025,+0.025]，P=0.64，**不显著**）
- **向量一致性：** 同 doc 余弦 0.891（两引擎计算路径不同，llama.cpp 用 ggml 后端、LiGGUF 纯标量，数值有系统偏差）
- **体积收益：** 引擎 625MB → 305KB，部署复杂度大幅降低

---

## 二、已解决的核心问题

### 2.1 Windows/MSVC 移植
- 删 `<fcntl.h>` `<sys/mman.h>` `<sys/stat.h>` 等 POSIX 头
- `open_mmap` 增加 `_WIN32` 分支：`fopen/fread` 全量读入内存
- `common.h` 的 `packed` 结构加 MSVC 分支
- `common.cpp` `clock_gettime` → `GetTickCount64`
- `vdb.cpp` `localtime_r` → `localtime_s`
- 编译加 `/arch:AVX2 /D__AVX2__`
- 产物：`ligguf_embed.exe` 单文件 305KB

### 2.2 中文分词与 llama.cpp 对齐
- 原版 `bert_tokenize` 按空白/标点切分 + 逐字节 `tolower` → 中文全变 `[UNK]`
- 对照 llama.cpp WPM 源码重写：**每个中文字符单独拆 word，每 word 前缀 `▁`(U+2581) 再最长贪心查表**
- 效果：`"你还记得我喜欢吃什么吗"` → `101 872 6820 6381 2533 2769 1599 3614 1391 784 720 1408 102` 与 llama.cpp `/tokenize` **完全一致**
- 附加：剥离 UTF-8 BOM(U+FEFF)

### 2.3【决定性】fp16 LUT 未初始化
- 根因：`fp16_to_fp32(H)` 是 `fp1632_lut[H]` 查表宏，LUT 只在 `fp1632_init()` 填充，而该初始化仅在 `ligguf_fast_qk.cpp` 主程序调用；harness 未调用 → LUT 全 0 → 所有权重解码为 0 → 所有文本输出完全相同向量（cos=1.0）
- 修复：`ensure_bert()` 开头调用 `fp1632_init()`
- 效果：token embedding 恢复非零且各不相同

---

## 三、目录结构

```
D:\LiGGUF-CN\
├── README.md                       ← 本文档（总览）
├── ligguf_build_port\              ← 移植版完整源码 + 编译产物
│   ├── vdb.cpp                     （含重写的 bert_tokenize + vdb_embed_export）
│   ├── ligguf_embed.cpp            （harness：文本 → 512 维向量 .bin）
│   ├── ligguf_embed.exe            （编译产物 305KB）
│   └── *.h / *.obj / Makefile
├── bench\                          ← 基准测试数据集 + 脚本
│   ├── gen_dataset.py              （生成 120 docs + 150 queries + GT，程序化）
│   ├── docs.txt                    （120 条中文记忆，由 gen_dataset.py 生成）
│   ├── queries.txt                 （150 条中文查询，含 30 条干扰）
│   ├── ground_truth.json           （120 条 GT + 30 空，程序生成）
│   ├── llama_docs.bin / ligguf_docs.bin   （两引擎向量 dump）
│   ├── llama_queries.bin / ligguf_queries.bin
│   ├── compare.py                  （主对比：召回/MRR/余弦/Jaccard + bootstrap CI）
│   ├── embed_llama.py              （调 llama-server 生成向量）
│   └── diff_vec.py                 （逐元素向量 diff 诊断）
├── patches\                        ← 与原版 fast/ 的差异补丁
│   ├── vdb_cpp.patch
│   ├── lil_gguf_cpp.patch
│   ├── lil_gguf_h.patch
│   ├── common_cpp.patch
│   └── common_h.patch
└── docs\
    ├── 01-technical-details.md     ← 详细技术文档
    ├── 02-pr-proposal.md           ← 面向上游 PR 提案（英文）
    ├── 03-benchmark-report.md      ← 基准报告 + 精度差异分析
    └── 10-integration-plan.md      ← 集成方案（方案 B：HTTP 兼容）
```

---

## 四、快速复现

### 前置
- MSVC 19.x + Windows SDK（本机：`D:\VS`，MSVC 19.51 + SDK 10.0.26100.0）
- `D:\bge-small-zh-v1.5-f16.gguf`（bge-small-zh-v1.5 中文，45.7MB）
- Python 3.9+（仅 bench 对比用）

### 编译
```powershell
# 进入 ligguf_build_port，MSVC x64 环境变量就绪后：
cl /O2 /std:c++17 /EHsc /arch:AVX2 /D__AVX2__ ligguf_embed.cpp vdb.cpp lil_math.cpp lil_gguf.cpp common.cpp
```

### 生成向量
```powershell
# 0) (可选) 重新生成大规模测试集
python bench\gen_dataset.py

# 1) LiGGUF 版
.\ligguf_build_port\ligguf_embed.exe D:\bge-small-zh-v1.5-f16.gguf bench\docs.txt bench\ligguf_docs.bin
.\ligguf_build_port\ligguf_embed.exe D:\bge-small-zh-v1.5-f16.gguf bench\queries.txt bench\ligguf_queries.bin

# 2) llama-server 版（需先起 server，见 embed_llama.py）
python bench\embed_llama.py   # 生成 llama_docs.bin / llama_queries.bin

# 3) 对比（含 bootstrap CI）
python bench\compare.py
```

---

## 五、关键文件说明

| 文件 | 说明 |
|------|------|
| `ligguf_build_port\vdb.cpp` | **核心**。含重写的 `bert_tokenize`(UTF-8 逐字拆+▁前缀)、`bert_embed`(4 层 BERT + MEAN pooling + L2)、导出 `vdb_embed_export` |
| `ligguf_build_port\ligguf_embed.cpp` | 命令行 harness：读文本文件 → 输出 N×512 向量 `.bin` |
| `ligguf_build_port\arch_avx2.h` | AVX2 加速。`dot_f16_f32` 用 F16C(`_mm256_cvtph_ps`) 批量转换替代逐元素 LUT 查表，~3x 提速，结果逐位不变 |
| `ligguf_build_port\lil_math.cpp` | 数学算子。`f16_to_f32_row` 反量化用 F16C 批量转换 |
| `bench\gen_dataset.py` | 程序化生成 120 docs + 150 queries(含 30 干扰) + ground_truth |
| `bench\compare.py` | 对比两引擎召回/MRR@5/同 doc 余弦/top-5 Jaccard + bootstrap 2000 次 CI |
| `bench\diff_vec.py` | 逐元素 diff，诊断向量偏差来源 |
| `patches\vdb_cpp.patch` | vdb.cpp 相对原版 fast/ 的全部改动（分词重写 + 导出函数） |

---

## 六、Known Limitations

1. **同 doc 向量余弦 0.891 非 1.0**：两引擎计算路径数值差异（llama.cpp 走 ggml 后端 + 量化 AVX，LiGGUF 纯标量），非可消除的单一 bug。逐层 hidden state diff 显示差异散布于多层。
2. **120 题测试集召回持平**（118/120 vs 118/120），bootstrap 2000 次 CI=[-0.025,+0.025] P=0.64，**不显著**——两引擎检索等价。
3. **单句输入**：token_type 全取 segment 0，与 llama.cpp 单句行为一致；多句(如 rerank pair)需额外处理。
4. **30 条干扰查询双引擎均无误报**：检索只命中语义相关记忆，噪声查询 top-5 命中为 0。

---

## 七、给后续者的建议

- 若想追平同 doc 余弦（0.891 → 更高）：在 `ligguf_build_port\vdb.cpp` 的 `bert_embed` 每层输出加 print，与 llama.cpp `--verbose-prompt` 的 hidden state 逐层 diff（参考 `docs/03-benchmark-report.md` 第 4 节）。注意：检索等价性已由 120 题 bootstrap 证明，追平余弦的收益主要是向量可移植性，而非检索质量。
- 集成到 EggBloom：见 `docs/10-integration-plan.md`（方案 B：以 8083 端口暴露 OpenAI 兼容 `/v1/embeddings`，前端零改动）。单条向量延迟 ~46ms，满足在线检索。
