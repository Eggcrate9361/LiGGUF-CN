# -*- coding: utf-8 -*-
import struct, json, math, io, sys, random, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = os.path.dirname(os.path.abspath(__file__))
docs = [l.strip() for l in open(base + r"\docs.txt", encoding="utf-8") if l.strip()]
queries = [l.strip() for l in open(base + r"\queries.txt", encoding="utf-8") if l.strip()]
gt = json.load(open(base + r"\ground_truth.json", encoding="utf-8"))

def load(fn):
    with open(fn, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        rest = f.read()
        dim = len(rest) // (n * 4)
        rows = []
        for i in range(n):
            raw = rest[i*dim*4 : (i+1)*dim*4]
            rows.append(struct.unpack("<%df" % dim, raw))
    return rows

lld, llq = load(base + r"\llama_docs.bin"), load(base + r"\llama_queries.bin")
lgd, lgq = load(base + r"\ligguf_docs.bin"), load(base + r"\ligguf_queries.bin")
print("llama docs", len(lld), "q", len(llq), "| ligguf docs", len(lgd), "q", len(lgq))
# docs.txt 末尾有空行,导致 ligguf 多了头部一条空文本向量;去掉它保持对齐
if len(lgd) > len(lld):
    lgd = lgd[:len(lld)]
if len(lgq) > len(llq):
    lgq = lgq[:len(llq)]

def norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n else v

lld = [norm(v) for v in lld]; llq = [norm(v) for v in llq]
lgd = [norm(v) for v in lgd]; lgq = [norm(v) for v in lgq]

def cos(a, b):
    return sum(x*y for x, y in zip(a, b))

def topk(q, db, k):
    s = [(cos(q, d), i) for i, d in enumerate(db)]
    s.sort(reverse=True)
    return [i for _, i in s[:k]], [c for c, _ in s[:k]]

print("\n=================== top-5 召回(命中 ground truth)===================")
K = 5
ll_total = lg_total = 0
ll_mrr = lg_mrr = 0.0
detail_l = []
for i, q in enumerate(queries):
    gt_idx = set(gt[i])
    tl, sl = topk(llq[i], lld, K)
    tg, sg = topk(lgq[i], lgd, K)
    hl = len(set(map(str, tl)) & gt_idx); hg = len(set(map(str, tg)) & gt_idx)
    ll_total += hl; lg_total += hg
    rl = min([1/(t+1) for t in tl if str(t) in gt_idx] or [0])
    rg = min([1/(t+1) for t in tg if str(t) in gt_idx] or [0])
    ll_mrr += rl; lg_mrr += rg
    print(f"Q{i:>2} {q[:24]:<26} llama={hl}  ligguf={hg}   MRR L={rl:.2f} G={rg:.2f}")

print(f"\ntop-5 命中总数: llama={ll_total}  ligguf={lg_total}")
print(f"MRR@5       : llama={ll_mrr/len(queries):.3f}  ligguf={lg_mrr/len(queries):.3f}")

print("\n=================== top-3 明细 ===================")
for i, q in enumerate(queries):
    gt_idx = set(gt[i])
    tl, _ = topk(llq[i], lld, 3)
    tg, _ = topk(lgq[i], lgd, 3)
    ll = " <-HIT" if str(tl[0]) in gt_idx else (" (in5)" if set(map(str,tl[1:])) & gt_idx else "")
    lgg = " <-HIT" if str(tg[0]) in gt_idx else (" (in5)" if set(map(str,tg[1:])) & gt_idx else "")
    lg_disp = []
    for x in tg:
        if x < len(docs):
            lg_disp.append(f"{x}:{docs[x][:14]}")
        else:
            lg_disp.append(f"{x}:(head)")
    print(f"Q{i:>2} {q[:20]} | llama: {[f'{x}:{docs[x][:14]}' for x in tl]}{ll}")
    print(f"       {'':<7} | ligguf:{lg_disp}{lgg}")

print("\n=================== 向量一致性 ===================")
same = [cos(lld[i], lgd[i]) for i in range(len(lld))]
print("同一 doc 的 llama·ligguf 余弦: 均值=%.3f 最小=%.3f 最大=%.3f" %
      (sum(same)/len(same), min(same), max(same)))
base_sim = []
for i in range(len(lld)):
    for j in range(len(lgd)):
        if i != j:
            base_sim.append(cos(lld[i], lgd[j]))
print("跨引擎不同 doc 平均余弦=%.3f (基数)" % (sum(base_sim)/len(base_sim)))

print("\n=================== top-5 集合 Jaccard ===================")
js = []
for i in range(len(queries)):
    tl = set(topk(llq[i], lld, 5)[0]); tg = set(topk(lgq[i], lgd, 5)[0])
    js.append(len(tl & tg) / len(tl | tg))
print("top-5 Jaccard 均值=%.2f (1=两个引擎完全一致)" % (sum(js)/len(js)))


def topk_indices(q, db, k):
    s = [(cos(q, d), i) for i, d in enumerate(db)]
    s.sort(reverse=True)
    return [i for _, i in s[:k]]


def hits_in5(q, db, gt_idx):
    tl = topk_indices(q, db, 5)
    return sum(1 for x in tl if str(x) in gt_idx)


def mrr(q, db, gt_idx):
    tl = topk_indices(q, db, 5)
    for rank, x in enumerate(tl, start=1):
        if str(x) in gt_idx:
            return 1.0 / rank
    return 0.0


print("\n=================== bootstrap CI (配对, 仅含 GT 的查询) ===================")
n_gt = [i for i in range(len(queries)) if gt[i]]
B = 2000
random.seed(2026)
pair_hits = [(hits_in5(llq[i], lld, set(gt[i])), hits_in5(lgq[i], lgd, set(gt[i]))) for i in n_gt]
pair_mrr = [(mrr(llq[i], lld, set(gt[i])), mrr(lgq[i], lgd, set(gt[i]))) for i in n_gt]
hit_diffs = []
mrr_diffs = []
for _ in range(B):
    rs = [random.randrange(len(n_gt)) for _ in range(len(n_gt))]
    hit_diffs.append(sum(pair_hits[r][1] - pair_hits[r][0] for r in rs) / len(rs))
    mrr_diffs.append(sum(pair_mrr[r][1] - pair_mrr[r][0] for r in rs) / len(rs))

hit_diffs.sort(); mrr_diffs.sort()
lo_h = hit_diffs[int(0.025*B)]; hi_h = hit_diffs[int(0.975*B)]
lo_m = mrr_diffs[int(0.025*B)]; hi_m = mrr_diffs[int(0.975*B)]
p_h = sum(1 for d in hit_diffs if d >= 0) / B
p_m = sum(1 for d in mrr_diffs if d >= 0) / B
print(f"n(带GT查询)={len(n_gt)}  bootstrap={B}")
print(f"命中数差(ligguf-llama): 均值={sum(pair_hits[i][1]-pair_hits[i][0] for i in range(len(n_gt)))/len(n_gt):+.4f}  95%CI=[{lo_h:+.4f},{hi_h:+.4f}]  P(差>=0)={p_h:.3f}")
print(f"MRR差  (ligguf-llama): 均值={sum(pair_mrr[i][1]-pair_mrr[i][0] for i in range(len(n_gt)))/len(n_gt):+.4f}  95%CI=[{lo_m:+.4f},{hi_m:+.4f}]  P(差>=0)={p_m:.3f}")
print("解读: 若 95%CI 不跨 0 且 P<0.05, 两引擎差异统计显著; 否则等价")