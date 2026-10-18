# -*- coding: utf-8 -*-
"""通过 gguf.GGUFReader 拿到 tokens[id] 的字面值"""
import gguf, sys, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

r = gguf.GGUFReader(r"D:\bge-small-zh-v1.5-f16.gguf")
# 调用内部 _get_str 是被验证可行的方法 (gguf-py 内部用)
# 我们用 _build_field 的方式：tokens 字段的 parts 是 [kv_klen, kv_kdata, raw_kv_type, array_header, blob...]
# gguf 库把所有 STRING array 元素的 strings 紧凑放在 field.data 字段里
# gguf.ReaderField 有 .data 属性 (tobytes()) 和 .offset字段
# 实际上 gguf.ReaderField 没有 .contents - contents 是方法

# 让我直接参考 _get_str: 它读 u64 len 然后 bytes
# tokens array 的 parts[3..] 是 n+1 个 u64 offset 和 n+1 个 u64 长度? 还是只有 offset?
# 简单粗暴：method 'data' 是 metadata 的 total bytes? 不确定，最快是按 field.data 的 np array 重新解析
# gguf 库实现里: type ARRAY 的 field, 有 parts[i] 是 array 元素的 data, parts[i+1] 是 lengths
# 看 ReaderField 实际是 namedtuple 有 fields: name, offset, parts, types, data, index, count, contents
# 看 gguf-py/gguf/gguf_reader.py 定义

# 直接遍历 fields 的 parts
f = r.fields['tokenizer.ggml.tokens']
print("field types:", f.types)
# 通常 parts = [len_part, str_part, type_part, header_part, off_array_part, blob_part]
# indices 指向 string 元素: types 是 [ARRAY, STRING], field.data 是字符串 blob 的 tobytes
# 字段元对象约定: ReaderField._make((name, offset, parts, types))
# gguf-py 文档示例直接写: for i, part in enumerate(field.parts); print(part.tobytes())

# 试拿 .data
print("dir:", [a for a in dir(r) if not a.startswith('_')][:30])
print("has _tokens?", hasattr(r, '_tokens'))

# 直接 try: 读 i-th token via internal method
# gguf 库 1.0+ 把每个 STRING 紧凑在 data 里; parts 是 list of ndarray, idx 是 index list
# 看 gguf-py/gguf/gguf_reader.py:
#   parts: list[NDArray[Any]]
#   types: list[GGUFValueType]
#   index: n dimensional index hint
# field.parts 列表里:
#   parts[0]: kv_klen (u32)
#   parts[1]: kv_kdata (bytes)
#   parts[2]: raw_kv_type (u32)
#   parts[3+]: field parts
# _get_field_parts 返回 field_size, field_parts, field_idxs, field_types
# 对 ARRAY of STRING:
#   field_parts 是 [array_header (u32 type, u64 n), offsets_arr (u64, n+1), char_blob_arr (u8)]
# 读完所有，self._data 不存
# 访问 i-th token: field.parts[-1][off[i]:off[i+1]].tobytes().decode('utf-8')

# 让我用 inspect 看 _get_field_parts
import inspect
print("----- _get_field_parts start ----")
try:
    print(inspect.getsource(r._get_field_parts))
except Exception as e:
    print("err:", e)
