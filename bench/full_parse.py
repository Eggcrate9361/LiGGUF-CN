import struct

path = r'D:\bge-small-zh-v1.5-f16.gguf'
d = open(path, 'rb').read()

o = 0
o += 4  # magic
ver = struct.unpack_from('<I', d, o)[0]; o += 4
nten = struct.unpack_from('<Q', d, o)[0]; o += 8
nkv = struct.unpack_from('<Q', d, o)[0]; o += 8

def rdstr(o):
    n = struct.unpack_from('<Q', d, o)[0]; o += 8
    return d[o:o+n].decode('utf-8', 'replace'), o+n

SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_val(o, t):
    if t in SIZE: return o + SIZE[t]
    if t == 8: return rdstr(o)[1]
    if t == 9:
        at = struct.unpack_from('<I', d, o)[0]; o += 4
        cnt = struct.unpack_from('<Q', d, o)[0]; o += 8
        for _ in range(cnt):
            o = skip_val(o, at)
        return o
    raise ValueError(t)

for i in range(nkv):
    k, o = rdstr(o)
    t = struct.unpack_from('<I', d, o)[0]; o += 4
    o = skip_val(o, t)

print('after kv @', o)
print('aligned32:', o + (32 - o % 32) % 32)

tensors = []
for i in range(nten):
    name, o = rdstr(o)
    ndim = struct.unpack_from('<I', d, o)[0]; o += 4
    dims = struct.unpack_from('<'+'Q'*ndim, d, o); o += 8*ndim
    typ = struct.unpack_from('<I', d, o)[0]; o += 4
    off = struct.unpack_from('<Q', d, o)[0]; o += 8
    tensors.append((name, dims, typ, off))
print('after tensor info @', o)
print('aligned32:', o + (32 - o % 32) % 32)
print()
for name, dims, typ, off in tensors[:8]:
    n = 1
    for dd in dims: n *= dd
    sz = n * 2 if typ == 1 else n * 4
    print(f'  {name:<40} dims={dims} type={typ} off={off} size={sz}')
print('...')
print('last tensor:', tensors[-1][0], 'off', tensors[-1][3])