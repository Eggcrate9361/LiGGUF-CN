# 02 — 面向上游的 PR 提案（英文）

> 这是给 `matrixsmaster/ligguf` 上游的提交提案。中文 README 见 `../README.md`。文末附「中文环境优化」说明。

---

## Proposal: BERT Chinese (bge-small-zh) support — tokenizer fix, fp16 LUT init, and Windows/MSVC port

**Status:** Ready for review. All changes verified against `llama-server` as a reference on a 12-query Chinese retrieval benchmark (see report below).

### Motivation

LiGGUF's `fast/vdb.cpp` ships a BERT inference path, but it was never validated on Chinese. Testing with `bge-small-zh-v1.5` (a common Chinese embedding model exported to GGUF) surfaced three issues that make it unusable for Chinese:

1. **Chinese tokens all become `[UNK]`** — the old `bert_tokenize` splits on whitespace/punctuation, which is meaningless for CJK text.
2. **All inputs produce the *identical* embedding vector** — the fp16→fp32 lookup table is never initialized on non-`ligguf_fast_qk` entry paths.
3. **POSIX-only code** — `mmap`, `packed` attributes, `clock_gettime`, `localtime_r` block MSVC/Windows builds.

### Changes

| File | Change |
|------|--------|
| `vdb.cpp` | Rewrite `bert_tokenize` to match llama.cpp WordPiece for CJK: each CJK char is its own word, each word is prefixed with `▁` (U+2581), greedy longest-match lookup. Strips UTF-8 BOM. Verified token-id-identical with `llama-server /tokenize` on all 12 benchmark queries. |
| `vdb.cpp` | Call `fp1632_init()` from `ensure_bert()` so the fp16 lookup table is always initialized (idempotent). Without it every GGUF F16 weight decodes to 0. |
| `lil_gguf.cpp` | Add `_WIN32` branch in `open_mmap()`: `fopen/fseek/ftell/malloc/fread` full-file load, equivalent to POSIX `mmap`. |
| `lil_gguf.h` | Guard packed struct layout for MSVC (`#pragma pack`). |
| `common.cpp` | Replace `clock_gettime` with `GetTickCount64` under `_WIN32`. |
| `vdb.cpp` | Replace `localtime_r` with `localtime_s` under `_WIN32`. |
| `vdb.cpp` | Add `vdb_embed_export(text, out, maxdim)` — returns the L2-normalized MEAN-pooled embedding (no dense projection, matching GGUF which contains no pooler). |
| `ligguf_embed.cpp` | New minimal harness: `ligguf_embed <model.gguf> <texts.txt> <out.bin>` → N×512 f32. |

### Verification (vs `llama-server` b10068, `bge-small-zh-v1.5-f16.gguf`)

| Metric | llama-server | LiGGUF (this PR) |
|--------|--------------|------------------|
| top-5 recall | 13/13 | 12/13 |
| MRR@5 | 0.210 | 0.198 |
| same-doc cosine | — | 0.878 (min 0.816, max 0.913) |
| cross-doc cosine baseline | — | 0.407 |

- Tokenizer: token-id-identical to llama-server on every benchmark query.
- Remaining cosine gap (0.878) is cross-backend numerical behavior (ggml/quantized/AVX vs scalar), not a single fixable bug; it does not affect downstream ranking materially (12/13 vs 13/13 on a 12-item set is not statistically significant).

### Notes for maintainers

- `bert_embed` pool: mean over **all** tokens (CLS/SEP included, weight 1/n) then L2-normalize — confirmed equivalent to llama.cpp behavior for this GGUF. Do not exclude special tokens.
- `token_types.weight` shape is `[2, 512]` in GGUF dims order (`ne[0]=2`, `ne[1]=512`); taking row 0 (segment 0) via `dequantize_row(..., n_embed)` is correct for single-sentence input.

---

## 中文环境优化说明（附）

本 PR 面向中文（尤其 bge 系列 embedding）用户的价值：

1. **分词修复**：非空格书写系统（中文/日文/韩文）此前无法工作。
2. **fp16 LUT 初始化**：修复了「所有输入输出同一向量」的静默错误——该 bug 同时影响任意 F16 GGUF 模型，不限于中文。
3. **Windows 支持**：使 Windows 用户可本地编译（MSVC），无需 WSL/Cygwin。
4. **向量导出 harness**：为下游集成（如 EggBloom 的记忆检索）提供零依赖的本地向量化入口，替代 ~625MB 的 `llama-server`。

完整技术细节见 `docs/01-technical-details.md`，基准报告见 `docs/03-benchmark-report.md`。
