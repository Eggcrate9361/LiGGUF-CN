# 10 — 集成方案（EggBloom）

> 当前状态：**移植 + 修复 + 基准验证已完成，集成搁置**。启用时按此文档操作。

---

## 方案 A：进程内嵌入（推荐，性能最优）

EggBloom 的检索服务以 `.dll` 方式加载本引擎，直接调 C 接口取向量，无进程开销。

**前置**：把 `vdb_embed_export` 所在源码编译为 DLL，导出：

```cpp
extern "C" __declspec(dllexport)
int vdb_embed_query(const char* model_path, const char* text, float* out, int maxdim);
```

**调用示意（EggBloom 侧，C#）：**

```csharp
[DllImport("ligguf_embed.dll")]
static extern int vdb_embed_query(string modelPath, string text, float[] outBuf, int maxdim);
```

**要点**：
- 模型一次性加载（`vdb_load` 缓存 gguf_model），多线程下用互斥锁串行化（当前实现非线程安全）
- 每 query 约几 ms（4 层 BERT，纯 CPU）
- 向量直接写入 EggBloom 的向量库，替换 llama-server 调用点

## 方案 B：HTTP 兼容层（改动最小）

在 `ligguf_embed.cpp` 基础上加一个 HTTP 服务器，暴露 OpenAI 兼容的 `/v1/embeddings`，**前端零改动**替换 8083 端口。

```
POST /v1/embeddings
{ "model": "bge-small-zh-v1.5", "input": "文本..." }
→ { "data": [ { "embedding": [0.123, ...512 floats...] } ] }
```

**要点**：
- 用 Windows `WinHTTP` 或轻量 lib（如 cpp-httplib 单头文件）即可，无需引入框架
- 保持 `/health` 探活，供 docker-compose 存活检查
- 模型加载一次驻留；每请求 `vdb_embed_export`
- 与 llama-server 的差异：仅支持单句 input（数组形式逐条串行处理即可）

## 方案 C：CLI 批处理（最简单，适合首次验证）

已实现：`ligguf_embed.exe <model.gguf> <texts.txt> <out.bin>`。适合一次性建库，不适合在线查询。

---

## 建议

- 首次接入用 **方案 C** 验证流程（半小时内跑通，直接产出全量文档向量）
- 长期用 **方案 B** 平滑替换（复用现有 llama-server 调用链，只换端口背后的进程）
- 若后续有性能诉求再上 **方案 A**

## 已知边界

- 单句输入（token_type 固定 segment 0）；多句 pair（如 rerank）需扩展 `bert_embed`
- 模型文件路径在 DLL 场景下建议传绝对路径，避免相对路径解析歧义
- 向量维数固定 512（bge-small）；换模型需同步改 `n_embed`（从 GGUF 读，已有）
