# -*- coding: utf-8 -*-
import json, urllib.request, struct, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def embed_many(texts):
    out = []
    for t in texts:
        payload = json.dumps({"input": t, "model": "bge-small-zh"}).encode('utf-8')
        req = urllib.request.Request("http://127.0.0.1:8083/v1/embeddings", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read().decode('utf-8'))
        out.append(d["data"][0]["embedding"])
    return out

base = os.path.dirname(os.path.abspath(__file__))
docs = [l.strip() for l in open(base + r"\docs.txt", encoding="utf-8") if l.strip()]
queries = [l.strip() for l in open(base + r"\queries.txt", encoding="utf-8") if l.strip()]
print("embedding docs:", len(docs))
de = embed_many(docs)
print("embedding queries:", len(queries))
qe = embed_many(queries)
print("dim:", len(de[0]))

def save(fn, vecs):
    with open(fn, "wb") as f:
        f.write(struct.pack("<i", len(vecs)))
        for v in vecs:
            f.write(struct.pack("<%df" % len(v), *v))
    print("saved", fn)

save(base + r"\llama_docs.bin", de)
save(base + r"\llama_queries.bin", qe)