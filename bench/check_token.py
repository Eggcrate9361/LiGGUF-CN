# -*- coding: utf-8 -*-
import gguf, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

r = gguf.GGUFReader(r"D:\bge-small-zh-v1.5-f16.gguf")
toks_field = r.fields['tokenizer.ggml.tokens']
# toks_field.parts 是 list[ndarray]; 大概 parts[0] 是 array header, parts[1:] 是每个 STRING 的 length+bytes
# 直接遍历: parts[1] 是 offset 数组 (n+1 个 u64), parts[2] 是去除 header 的 char 数组
# 更稳的方法: gguf 库提供 ReaderField.value 或 .parts[-1].tobytes()
print("n parts:", len(toks_field.parts), "types:", toks_field.types)
for i, part in enumerate(toks_field.parts[:5]):
    print(f"  part[{i}] shape={part.shape} dtype={part.dtype} bytes={part.nbytes}")
# 通常 parts[0]: array header (u32 etype, u64 n); parts[1]: offsets[n+1] u64; parts[2]: char blob
off = toks_field.parts[1]
blob = toks_field.parts[2]
n = off.shape[0] - 1
print(f"n_tokens = {n}")
tokens = []
for i in range(n):
    s = off[i]; e = off[i+1]
    tokens.append(bytes(blob[s:e]).decode('utf-8', 'replace'))

# 现在 tokens[872] 应该是 '▁你', tokens[13929] 应该是 '你' (裸字)
for tid in [101, 102, 872, 13929, 6820, 19877, 6381, 19438, 1391, 4500, 2787, 1599, 3614]:
    if tid < len(tokens):
        b = tokens[tid].encode('utf-8')
        print(f"  tokens[{tid:>5}] = {tokens[tid]!r}  bytes={[hex(x) for x in b]}")
    else:
        print(f"  tokens[{tid:>5}] OOB len(tokens)={len(tokens)}")
