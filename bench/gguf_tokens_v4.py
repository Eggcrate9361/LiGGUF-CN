# -*- coding: utf-8 -*-
"""通过 gguf.GGUFReader 读 tokenizer.ggml.tokens 数组, 拿到 tokens[id] 字面值"""
import gguf, sys, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

r = gguf.GGUFReader(r"D:\bge-small-zh-v1.5-f16.gguf")
f = r.fields['tokenizer.ggml.tokens']

# parts: [kv_klen, kv_kdata, raw_kv_type, *(flatten_elements)]
# 对 ARRAY: 第 3 part 是 raw_itype (u32), 第 4 part 是 alen (u64), 之后是每个元素的 _get_str 返回值 = [u64_len, u1_bytes]
# _get_str 返回: tuple of 2 ndarrays: len_arr (u64) and byte_arr (u8)
# 所以 parts[3..] = [itype, alen, len_i0, bytes_i0, len_i1, bytes_i1, ...]
parts = f.parts
print("n parts:", len(parts))
# 找出 alen 在哪
itype = parts[3]  # u32
alen = parts[4]   # u64
print(f"itype={itype} alen={alen} (so {alen[0]} tokens)")
n = int(alen[0])
print("parts[5] shape:", parts[5].shape, "dtype:", parts[5].dtype)
print("parts[6] shape:", parts[6].shape, "dtype:", parts[6].dtype)
# parts[5] = u64 长度 of token 0
# parts[6] = u8 bytes of token 0
# 应按 idx 各占 2 parts. 但 _get_field_parts for ARRAY 的 idx 添加: aparts += curr_parts (即 [len, byte]); data_idxs 加 idx + idxs_offs.
# 所以总的: parts = [kv_klen, kv_kdata, raw_kv_type, raw_itype, alen, *flatten]
# parts[5+2*i] = i-th token 的 len (u64); parts[5+2*i+1] = i-th token 的 bytes (uint8)
# 验证: parts[5] 期望 (1,) u64; parts[6] 期望 (x,) u8

def token_at(i):
    # 防御性 size 检查
    if 5 + 2*i + 1 >= len(parts):
        return None
    nlen_arr = parts[5 + 2*i]
    bytes_arr = parts[5 + 2*i + 1]
    if nlen_arr.shape != (1,) or bytes_arr.dtype != np.uint8:
        return None
    nlen = int(nlen_arr[0])
    if bytes_arr.size < nlen:
        return None
    b = bytes(bytes_arr[:nlen].tobytes())
    return b.decode('utf-8', 'replace')

# 验证 tokens[0..3]
for i in [0, 1, 2, 3, 100, 101, 102, 103]:
    print(f"tokens[{i}] = {token_at(i)!r}")

print("\n--- 关键 id 对比 ---")
for i in [101, 102, 100, 872, 13929, 6820, 19877, 6381, 19438, 1391, 4500, 2787, 1599, 3614, 4125, 7222, 117, 2215, 1071]:
    t = token_at(i)
    if t is not None:
        b = t.encode('utf-8')
        print(f"tokens[{i:>5}] = {t!r}  bytes={[hex(x) for x in b]}")
    else:
        print(f"tokens[{i:>5}] OOB")
