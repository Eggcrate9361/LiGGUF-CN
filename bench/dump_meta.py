import struct, sys

path = sys.argv[1]
with open(path, 'rb') as f:
    data = f.read()

off = 0
magic = data[off:off+4]; off += 4
assert magic == b'GGUF', magic
version = struct.unpack('<I', data[off:off+4])[0]; off += 4
n_tensors = struct.unpack('<Q', data[off:off+8])[0]; off += 8
n_kv = struct.unpack('<Q', data[off:off+8])[0]; off += 8
print(f"version={version} n_tensors={n_tensors} n_kv={n_kv}")

def rd_str(d, o):
    n = struct.unpack_from('<Q', d, o)[0]; o += 8
    return d[o:o+n].decode('utf-8', 'replace'), o+n

SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1}

def rd_val(d, o, t):
    if t in SIZES:
        return d[o:o+SIZES[t]], o+SIZES[t]
    if t == 8:
        return rd_str(d, o)
    raise ValueError(t)

for i in range(n_kv):
    k, off = rd_str(data, off)
    t = data[off]; off += 1
    if t in SIZES:
        _, off = rd_val(data, off, t)
    elif t == 8:
        _, off = rd_str(data, off)
    elif t == 9:
        at = data[off]; off += 1
        cnt = struct.unpack_from('<Q', data, off)[0]; off += 8
        for _ in range(cnt):
            _, off = rd_val(data, off, at)
    else:
        print("unknown type", t); sys.exit(1)
    print(f"seen KV[{i}] key='{k}' type={t} off_after={off}")