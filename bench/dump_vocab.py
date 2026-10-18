# -*- coding: utf-8 -*-
import struct, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
path = r"D:\bge-small-zh-v1.5-f16.gguf"
data = open(path, 'rb').read()
TYPE = {0:"U8",1:"I8",2:"U16",3:"I16",4:"U32",5:"I32",6:"F32",7:"BOOL",8:"STRING",9:"ARRAY",10:"U64",11:"I64",12:"F64"}

def rd32(p): return struct.unpack_from('<I', data, p)[0], p+4
def rd64(p): return struct.unpack_from('<Q', data, p)[0], p+8
def rdstr(p):
    ln, p = rd64(p)
    return data[p:p+ln].decode('utf-8','replace'), p+ln

def skipper(t, p):
    if t in (0, 7): return p + 1
    if t in (1, 2): return p + 2
    if t in (3, 4, 5, 6): return p + 4
    if t == 8:
        ln, p = rd64(p); return p + ln
    if t == 9:
        et, p = rd32(p); ne, p = rd64(p)
        for _ in range(ne): p = skipper(et, p)
        return p
    if t in (10, 11, 12): return p + 8
    return p

p = 4
ver, p = rd32(p)
nten, p = rd64(p); nmeta, p = rd64(p)
offs = {}
for _ in range(nmeta):
    k, p = rdstr(p)
    tag, p = rd32(p)
    offs[k] = (tag, p)
    p = skipper(tag, p)

tag, off = offs["tokenizer.ggml.tokens"]
p = off
et, p = rd32(p)
ne, p = rd64(p)
tokens = []
for _ in range(ne):
    s, p = rdstr(p)
    tokens.append(s)
print("vocab size:", ne)

llama_ids = [101, 872, 6820, 6381, 2533, 2769, 1599, 3614, 1391, 784, 720, 1408, 102]
ligguf_ids = [101, 13929, 19877, 19438, 15590, 15826, 14656, 16671, 14448, 13841, 13777, 14465, 102]
print("\nllama :", [f"{i}={tokens[i]!r}" for i in llama_ids])
print("ligguf:", [f"{i}={tokens[i]!r}" for i in ligguf_ids])

for ch in "你还记得我喜欢吃什么吗玩猫旅游作前音鹏":
    hits = [i for i, t in enumerate(tokens) if t == ch][:5]
    hits2 = [i for i, t in enumerate(tokens) if t == "##" + ch][:5]
    print(f"'{ch}' -> plain={hits} ##={hits2}")