# -*- coding: utf-8 -*-
"""
离线向量建库脚本（方案 C）
用 LiGGUF 的 ligguf_embed.exe 为 super_memory.db 中未向量化的 facts/events 批量生成 embedding 并写回。

用法：
  python build_super_memory_db.py <super_memory.db> [--model <model.gguf>] [--exe <ligguf_embed.exe>] [--dry-run]

默认路径：
  model: D:/bge-small-zh-v1.5-f16.gguf
  exe  : D:/LiGGUF-CN/ligguf_build_port/ligguf_embed.exe

流程：
  1) 读 facts/events 中 embedding 为空的条目（text 取 content/summary）
  2) 写 texts.txt -> 调 ligguf_embed.exe 批量出向量
  3) 解析 Nx512 float32 .bin，写回 embedding 列（json.dumps 列表，与运行时 _Embedder 格式一致）
"""
import os
import sys
import json
import struct
import sqlite3
import tempfile
import subprocess
import argparse


def load_vecs(bin_path: str):
    """解析 ligguf_embed.exe 输出 .bin：头 4 字节 int32 条数 + N*512 float32。"""
    with open(bin_path, "rb") as f:
        n = struct.unpack("<i", f.read(4))[0]
        rest = f.read()
        dim = len(rest) // (n * 4) if n else 0
        vecs = []
        for i in range(n):
            raw = rest[i * dim * 4: (i + 1) * dim * 4]
            vecs.append(struct.unpack("<%df" % dim, raw))
    return vecs


def build(db_path: str, model: str, exe: str, dry_run: bool):
    if not os.path.exists(db_path):
        print(f"DB 不存在: {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    targets = []  # (table, id, text)
    for table, col in (("facts", "content"), ("events", "summary")):
        rows = conn.execute(
            f'SELECT id, {col} FROM {table} WHERE embedding IS NULL OR embedding = ""'
        ).fetchall()
        for rid, text in rows:
            text = (text or "").strip()
            if text:
                targets.append((table, rid, text))
    print(f"待向量化条目: facts+events 共 {len(targets)}")

    if not targets:
        conn.close()
        print("无需建库，全部已向量化。")
        return 0

    if dry_run:
        print("[dry-run] 示例条目:")
        for table, rid, text in targets[:5]:
            print(f"  {table}#{rid}: {text[:40]}")
        conn.close()
        return 0

    tmp = tempfile.mkdtemp(prefix="ligguf_build_")
    texts_fn = os.path.join(tmp, "texts.txt")
    out_fn = os.path.join(tmp, "out.bin")
    with open(texts_fn, "w", encoding="utf-8") as f:
        for _, _, text in targets:
            f.write(text + "\n")

    print(f"运行: {exe}")
    r = subprocess.run([exe, model, texts_fn, out_fn], capture_output=True)
    sys.stdout.write(r.stdout.decode("utf-8", "replace"))
    sys.stderr.write(r.stderr.decode("utf-8", "replace"))
    if r.returncode != 0 or not os.path.exists(out_fn):
        print(f"ligguf_embed 失败 rc={r.returncode}")
        return 1

    vecs = load_vecs(out_fn)
    if len(vecs) < len(targets):
        print(f"向量数不足: {len(vecs)} < {len(targets)}")
        return 1
    vecs = vecs[: len(targets)]

    updated = 0
    for (table, rid, _), vec in zip(targets, vecs):
        emb = json.dumps([round(float(v), 6) for v in vec])
        conn.execute(f"UPDATE {table} SET embedding=? WHERE id=?", (emb, rid))
        updated += 1
    conn.commit()
    conn.close()
    print(f"写回完成: {updated} 条")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path")
    ap.add_argument("--model", default=r"D:/bge-small-zh-v1.5-f16.gguf")
    ap.add_argument("--exe", default=r"D:/LiGGUF-CN/ligguf_build_port/ligguf_embed.exe")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(build(a.db_path, a.model, a.exe, a.dry_run))


if __name__ == "__main__":
    main()
