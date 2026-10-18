# -*- coding: utf-8 -*-
import struct, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

data = open(r"D:\bge-small-zh-v1.5-f16.gguf", "rb").read()

p = 4
ver = struct.unpack_from('<I', data, p)[0]; p += 4
nten = struct.unpack_from('<Q', data, p)[0]; p += 8
nmeta = struct.unpack_from('<Q', data, p)[0]; p += 8
print(f"GGUF v{ver} n_tensors={nten} n_kv={nmeta}")

def rdstr(p):
    ln = struct.unpack_from('<Q', data, p)[0]; p += 8
    return data[p:p+ln].decode('utf-8', 'replace'), p+ln

SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

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

# 先读 KV
tensor_tinfos = []
for _ in range(nmeta):
    k, p = rdstr(p)
    tag = data[p]; p += 1
    print(f"KV: '{k}' type={tag}")
    p = sk(tag, p)

# 然后是 tensor info: 每个 tensor 是 name(str) + n_dims(u32) + dims(u64*n_dims) + offset(u64) + type(u32) + ???
print("\n--- tensor infos ---")
for i in range(nten):
    name, p = rdstr(p)
    ndims = struct.unpack_from('<I', data, p)[0]; p += 4
    dims = []
    for _ in range(ndims):
        d = struct.unpack_from('<Q', data, p)[0]; p += 8
        dims.append(d)
    off = struct.unpack_from('<Q', data, p)[0]; p += 8
    ttype = struct.unpack_from('<I', data, p)[0]; p += 4
    print(f"  T{i:>2} '{name}' dims={dims} off={off} type={ttype}")
