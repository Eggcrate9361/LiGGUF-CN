# -*- coding: utf-8 -*-
"""逐元素 diff: 同一 doc 在 llama vs ligguf 的向量对比, 找偏差来源"""
import struct, math, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r"C:\Users\ADMINI~1\AppData\Local\Temp\opencode\bench"

def load(fn):
    with open(fn, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        rest = f.read()
        dim = len(rest) // (n * 4)
        rows = []
        for i in range(n):
            raw = rest[i*dim*4:(i+1)*dim*4]
            rows.append(list(struct.unpack("<%df" % dim, raw)))
    return rows, dim

def l2norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x / n for x in v] if n else v

lld, dim = load(base + r"\llama_docs.bin")
lgd, _   = load(base + r"\ligguf_docs.bin")
print(f"llama docs={len(lld)} dim={dim}  ligguf docs={len(lgd)}")
N = min(len(lld), len(lgd))   # 去 ligguf 末尾的可能空行多余项

docs = [l.strip() for l in open(base + r"\docs.txt", encoding="utf-8") if l.strip()]

print("\n=================== 1. 同 doc 向量的整体 cos(未 norm) / cos(L2 后) ===================")
print(f"{'i':>3} {'|cos unnorm':>14} {'|cos L2':>10} {'|L2 llama':>10} {'|L2 ligguf':>10}  text")
for i in range(N):
    ll = lld[i]; lg = lgd[i]
    cn = sum(a*b for a, b in zip(ll, lg))
    l2l = math.sqrt(sum(x*x for x in ll))
    l2g = math.sqrt(sum(x*x for x in lg))
    lg_n = l2norm(lg); ll_n = l2norm(ll)
    cn_l2 = sum(a*b for a, b in zip(ll_n, lg_n))
    print(f"{i:>3} {cn:>14.4f} {cn_l2:>10.4f} {l2l:>10.4f} {l2g:>10.4f}  {docs[i][:30]}")

print("\n=================== 2. 逐维度 diff 看是否系统性偏(取前 3 条 doc) ===================")
for i in range(min(3, N)):
    ll = l2norm(lld[i]); lg = l2norm(lgd[i])
    print(f"\n-- doc {i}: {docs[i][:30]} (L2 norm 后对比 -- 应该完全一致才能召回一致)")
    diffs = [(k, lg[k]-ll[k]) for k in range(dim)]
    abs_d = sorted([abs(d) for _, d in diffs], reverse=True)
    print(f"  绝对差最大 5 维: {abs_d[:5]}")
    print(f"  绝对差均值: {sum(abs_d)/len(abs_d):.5f}")
    # 前若干维直接打印
    print(f"  llama 前 8 维: {[round(x,4) for x in ll[:8]]}")
    print(f"  ligguf 前 8 维: {[round(x,4) for x in lg[:8]]}")
    print(f"  diff  前 8 维: {[round(x,4) for x in [lg[k]-ll[k] for k in range(8)]]}")
    print(f"  llama 末 8 维: {[round(x,4) for x in ll[-8:]]}")
    print(f"  ligguf 末 8 维: {[round(x,4) for x in lg[-8:]]}")
    print(f"  diff  末 8 维: {[round(x,4) for x in [lg[k]-ll[k] for k in range(dim-8, dim)]]}")

print("\n=================== 3. 是否存在 'scale' 偏移(整体放大缩小) ===================")
# 若 LiGGUF 缺某 norm, 同 doc 两向量应近似比例关系: lg ≈ k * ll + b
# 拟合最小二乘 lg = k*ll (无截距), 看 k 是否一致
import statistics
for i in range(min(5, N)):
    ll = lld[i]; lg = lgd[i]
    num = sum(a*b for a, b in zip(ll, lg))
    den = sum(a*a for a in ll)
    k = num / den if den else 0
    # 残差
    resid = [lg[j] - k*ll[j] for j in range(dim)]
    rms = math.sqrt(sum(x*x for x in resid) / len(resid))
    print(f"doc {i}: k(=lg/ll 比例)={k:.5f}  残差 rms={rms:.5f}  llama|L2|={math.sqrt(den):.4f}  ligguf|L2|={math.sqrt(sum(x*x for x in lg)):.4f}")

print("\n=================== 4. 检查 LiGGUF 是否末尾未 L2 norm ===================")
# vdb.cpp:324 末尾有 l2norm(out); 但若未生效, ||lg|| 应非 1
for i in range(min(5, N)):
    l2 = math.sqrt(sum(x*x for x in lgd[i]))
    print(f"doc {i}: ||ligguf||={l2:.5f}")
