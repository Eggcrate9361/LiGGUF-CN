# -*- coding: utf-8 -*-
"""读 GGUF tensor info 列所有 tensor name (看看是否有 cls.dense / pooler / bert.cls 等dense projection 权重)"""
import struct, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

data = open(r"D:\bge-small-zh-v1.5-f16.gguf", "rb").read()
p = 4
ver = struct.unpack_from('<I', data, p)[0]; p += 4
nten = struct.unpack_from('<Q', data, p)[0]; p += 8
nmeta = struct.unpack_from('<Q', data, p)[0]; p += 8
print(f"GGUF v{ver} n_tensors={nten} n_kv={nmeta}")

to_data_off = p + nmeta * 0  # 待会算

# 完整解析 KV 才能跳到 tensor info
SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def rdstr(p):
    ln = struct.unpack_from('<Q', data, p)[0]; p += 8
    return data[p:p+ln].decode('utf-8', 'replace'), p+ln

def sk(t, p):
    if t in SIZE: return p + SIZE[t]
    if t == 8:
        _, p = rdstr(p); return p
    if t == 9:
        et = data[p]; p += 1
        ne = struct.unpack_from('<Q', data, p)[0]; p += 8
        for _ in range(ne): p = sk(et, p)
        return p
    raise ValueError(t)

kv_seen = []
for _ in range(nmeta):
    k, p = rdstr(p)
    tag = data[p]; p += 1
    # 读值
    val = None
    if tag in (0,1,2,3,4,5,6,7,10,11,12):
        v = data[p:p+SIZE[tag]]; p += SIZE[tag]
        if tag in (4,5): val = struct.unpack('<I', v)[0]
        elif tag in (6,): val = struct.unpack('<f', v)[0]
        elif tag in (10,11,12): val = struct.unpack('<Q', v)[0] if tag in (10,11) else struct.unpack('<d', v)[0]
        elif tag == 7: val = v[0]
        elif tag in (2,3): val = struct.unpack('<H', v)[0]
        elif tag in (0,1): val = v[0]
    elif tag == 8:
        val, p = rdstr(p)
    elif tag == 9:
        et = data[p]; p += 1
        ne = struct.unpack_from('<Q', data, p)[0]; p += 8
        arr = []
        for _ in range(ne):
            if et in SIZE:
                v = data[p:p+SIZE[et]]; p += SIZE[et]
                arr.append(v)
            elif et == 8:
                s, p = rdstr(p); arr.append(s)
        val = f"<array type={et} n={ne}>"
    kv_seen.append((k, tag, val))

for k, t, v in kv_seen:
    short = str(v)[:60]
    print(f"  KV: '{k}' t={t} v={short}")

print(f"\n--- {nten} tensors ---")
for i in range(nten):
    name, p = rdstr(p)
    ndims = struct.unpack_from('<I', data, p)[0]; p += 4
    dims = []
    for _ in range(ndims):
        d = struct.unpack_from('<Q', data, p)[0]; p += 8
        dims.append(d)
    # 下一项 offset(u64) 或 scale/... 但通常 n_dims + dims 之后是 type(u32) + offset(u64)?
    # gguf spec: name(str) + n_dims(u32) + dims(u64*n_dims) + type(u32) + offset(u64)
    ttype = struct.unpack_from('<I', data, p)[0]; p += 4
    off = struct.unpack_from('<Q', data, p)[0]; p += 8
    print(f"  T{i:>2} '{name}' dims={dims} type={ttype} off={off}")
