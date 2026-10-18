# -*- coding: utf-8 -*-
import gguf, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

r = gguf.GGUFReader(r"D:\bge-small-zh-v1.5-f16.gguf")
print(f"=== KV ({len(r.fields)} entries) ===")
for k, v in r.fields.items():
    print(f"  '{k}'  types={[str(t) for t in v.types]}  len={len(v.parts)}")
print(f"\n=== Tensors ({len(r.tensors)}) ===")
for t in r.tensors:
    print(f"  '{t.name}' shape={t.shape} type={t.tensor_type}")
