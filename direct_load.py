"""Load a trust_remote_code HF model straight onto the GPU, tensor by tensor, with parallel reads.

transformers' from_pretrained materializes the whole 9.3B-param Thinker in CPU RAM first (~14.6 GiB
peak), which under WSL's RAM cap pushes ~3 GB into swap on every start (needless SSD writes).
Here the module tree is built directly on the GPU and each checkpoint tensor is read and copied
into its parameter, so host RAM stays small. Reads are split across threads by file offset: the
WSL virtual disk gives ~0.5 GB/s to one reader but ~2.2 GB/s to eight.
"""
import json
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
       "BOOL": torch.bool}


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    return 8 + n, meta


def prefetch(paths, workers=8, chunk=64 << 20):
    """Pull files into the page cache with parallel reads (cheap way to speed up a later sequential load)."""
    jobs = []
    for p in paths:
        size = os.path.getsize(p)
        jobs += [(p, off, min(chunk, size - off)) for off in range(0, size, chunk)]

    def read(job):
        p, off, n = job
        fd = os.open(p, os.O_RDONLY)
        try:
            os.pread(fd, n, off)
        finally:
            os.close(fd)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(read, jobs))


def load_remote_model_direct(model_dir, device="cuda:0", dtype=torch.bfloat16, attn_impl="sdpa", workers=8,
                             special=None):
    """special: optional fn(model) -> {checkpoint_key: fn(cpu_tensor)} for tensors that need custom handling
    (e.g. quantize-on-load); those keys are routed to the hook instead of a parameter."""
    from transformers import AutoConfig, GenerationConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    for c in (config, getattr(config, "text_config", None)):
        if c is not None:
            c._attn_implementation = attn_impl
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], str(model_dir))

    t0 = time.time()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = cls(config)  # params that the code pins to float32 (A_log, dt_bias) stay float32
    finally:
        torch.set_default_dtype(prev)
    # The constructor queued random-init kernels on the default stream; the loader threads write on
    # their own streams, so wait for init to finish or it would overwrite the loaded weights.
    torch.cuda.synchronize(device)
    t1 = time.time()

    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    hooks = special(model) if special is not None else {}
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    jobs = []  # (file, abs_offset, nbytes, key, dtype, shape)
    unexpected = []
    for fname in sorted(set(index.values())):
        path = str(model_dir / fname)
        base, meta = _header(path)
        for k, info in meta.items():
            if k not in targets and k not in hooks:
                unexpected.append(k)
                continue
            a, b = info["data_offsets"]
            jobs.append((path, base + a, b - a, k, _DT[info["dtype"]], info["shape"]))
    jobs.sort(key=lambda j: (j[0], j[1]))
    # contiguous byte ranges per worker
    total = sum(j[2] for j in jobs)
    groups, cur, acc = [], [], 0
    for j in jobs:
        cur.append(j)
        acc += j[2]
        if acc >= total / workers and len(groups) < workers - 1:
            groups.append(cur)
            cur, acc = [], 0
    if cur:
        groups.append(cur)

    loaded = set()
    lock = threading.Lock()
    dev = torch.device(device)

    def work(group):
        stream = torch.cuda.Stream(device=dev)
        fds = {}
        try:
            with torch.no_grad(), torch.cuda.stream(stream):
                for path, off, n, k, dt, shape in group:
                    fd = fds.get(path)
                    if fd is None:
                        fd = fds[path] = os.open(path, os.O_RDONLY)
                    buf = bytearray(n)
                    mv = memoryview(buf)
                    got = 0
                    while got < n:
                        got += os.preadv(fd, [mv[got:]], off + got)
                    src = torch.frombuffer(buf, dtype=dt).reshape(shape) if n else torch.empty(shape, dtype=dt)
                    if k in hooks:
                        hooks[k](src)
                        with lock:
                            loaded.add(k)
                        continue
                    dst = targets[k]
                    if tuple(src.shape) != tuple(dst.shape):
                        raise ValueError(f"shape mismatch for {k}: {tuple(src.shape)} vs {tuple(dst.shape)}")
                    dst.copy_(src)  # synchronous H2D from pageable memory; casts dtype if needed
                    with lock:
                        loaded.add(k)
            stream.synchronize()
        finally:
            for fd in fds.values():
                os.close(fd)

    with ThreadPoolExecutor(len(groups)) as ex:
        for f in [ex.submit(work, g) for g in groups]:
            f.result()
    torch.cuda.synchronize(device)
    missing = [k for k, p in model.named_parameters() if k not in loaded and p.device.type != "meta"]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters missing from checkpoint, e.g. {missing[:5]}")
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    except Exception:
        pass
    model.eval()
    torch.cuda.empty_cache()
    t2 = time.time()
    info = dict(build_s=round(t1 - t0, 1), load_s=round(t2 - t1, 1), gb=round(total / 2**30, 1),
                unexpected=len(unexpected), unexpected_examples=unexpected[:5])
    return model, info
