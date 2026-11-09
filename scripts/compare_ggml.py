"""Compare two whisper.cpp GGML files: header, mel filters, vocabulary, then every tensor.

    .venv/bin/python -I scripts/compare_ggml.py a.bin b.bin

Tensors are matched by name, so a different tensor order does not count as a difference.
"""

import struct
import sys
from pathlib import Path

import numpy as np

HPARAMS = [
    "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
    "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels", "ftype",
]
DTYPES = {0: np.float32, 1: np.float16}


def parse(path: Path):
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    off = 0

    def ints(n):
        nonlocal off
        v = struct.unpack_from(f"<{n}i", buf, off)
        off += 4 * n
        return v

    (magic,) = ints(1)
    assert magic == 0x67676D6C, f"{path}: bad magic {magic:#x}"
    hp = dict(zip(HPARAMS, ints(len(HPARAMS))))
    n_mel, n_fft = ints(2)
    mel = np.frombuffer(buf, dtype="<f4", count=n_mel * n_fft, offset=off).reshape(n_mel, n_fft)
    off += 4 * n_mel * n_fft
    (n_tok,) = ints(1)
    vocab = []
    for _ in range(n_tok):
        (ln,) = ints(1)
        vocab.append(bytes(buf[off:off + ln]))
        off += ln
    header_end = off
    tensors = {}
    order = []
    while off < len(buf):
        n_dims, name_len, ftype = ints(3)
        shape = ints(n_dims)  # ggml order: innermost first
        name = bytes(buf[off:off + name_len]).decode()
        off += name_len
        dt = DTYPES[ftype]
        count = int(np.prod(shape))
        data = np.frombuffer(buf, dtype=dt, count=count, offset=off)
        off += count * np.dtype(dt).itemsize
        tensors[name] = (ftype, tuple(reversed(shape)), data)
        order.append(name)
    return hp, mel, vocab, header_end, tensors, order


def main():
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = parse(a_path), parse(b_path)
    ha, mel_a, voc_a, hend_a, ten_a, ord_a = a
    hb, mel_b, voc_b, hend_b, ten_b, ord_b = b
    print(f"A = {a_path.name} ({a_path.stat().st_size} bytes, header {hend_a} bytes)")
    print(f"B = {b_path.name} ({b_path.stat().st_size} bytes, header {hend_b} bytes)")

    print("\n## hparams")
    for k in HPARAMS:
        mark = "" if ha[k] == hb[k] else "   <-- differs"
        print(f"  {k:14s} A={ha[k]:<8d} B={hb[k]:<8d}{mark}")

    print("\n## mel filters")
    if mel_a.shape != mel_b.shape:
        print(f"  shape differs: A={mel_a.shape} B={mel_b.shape}")
    else:
        print(f"  shape {mel_a.shape}, identical={np.array_equal(mel_a, mel_b)}, "
              f"max|diff|={float(np.max(np.abs(mel_a - mel_b))):.3g}")

    print("\n## vocabulary")
    print(f"  A: {len(voc_a)} tokens, {sum(map(len, voc_a))} bytes of token text")
    print(f"  B: {len(voc_b)} tokens, {sum(map(len, voc_b))} bytes of token text")
    common = min(len(voc_a), len(voc_b))
    diff_ids = [i for i in range(common) if voc_a[i] != voc_b[i]]
    print(f"  first {common} ids: {common - len(diff_ids)} identical, {len(diff_ids)} differ")
    for i in diff_ids[:5]:
        print(f"    id {i}: A={voc_a[i]!r} B={voc_b[i]!r}")
    for name, voc, n in (("A", voc_a, len(voc_a)), ("B", voc_b, len(voc_b))):
        if n > common:
            extra = voc[common:]
            nonempty = [t for t in extra if t]
            print(f"  {name} has {n - common} extra ids ({common}..{n - 1}), "
                  f"{len(nonempty)} non-empty, e.g. {nonempty[:3]} ... {nonempty[-2:]}")

    print("\n## tensors")
    print(f"  A: {len(ten_a)} tensors, B: {len(ten_b)} tensors, same order: {ord_a == ord_b}")
    only_a = sorted(set(ten_a) - set(ten_b))
    only_b = sorted(set(ten_b) - set(ten_a))
    if only_a:
        print(f"  only in A: {only_a}")
    if only_b:
        print(f"  only in B: {only_b}")
    n_same = 0
    worst = (0.0, None)
    problems = []
    for name in ord_a:
        if name not in ten_b:
            continue
        fa, sa, da = ten_a[name]
        fb, sb, db = ten_b[name]
        if sa != sb or fa != fb:
            problems.append(f"  {name}: A {sa} ftype {fa}  B {sb} ftype {fb}")
            continue
        if da.tobytes() == db.tobytes():
            n_same += 1
            continue
        d = float(np.max(np.abs(da.astype(np.float32) - db.astype(np.float32))))
        n_diff = int(np.count_nonzero(da != db))
        problems.append(f"  {name} {sa}: {n_diff}/{da.size} values differ, max|diff|={d:.3g}")
        if d > worst[0]:
            worst = (d, name)
    print(f"  bit-identical tensors: {n_same}/{len(ten_a)}")
    for p in problems:
        print(p)
    if worst[1]:
        print(f"  worst: {worst[1]} max|diff|={worst[0]:.3g}")

    payload_a = sum(t[2].nbytes for t in ten_a.values())
    payload_b = sum(t[2].nbytes for t in ten_b.values())
    print(f"\n  tensor payload bytes: A={payload_a} B={payload_b}")
    print(f"  file size difference: {a_path.stat().st_size - b_path.stat().st_size} bytes, "
          f"header difference: {hend_a - hend_b} bytes")


if __name__ == "__main__":
    main()
