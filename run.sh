#!/usr/bin/env bash
cd "$(dirname "$0")"
export HSA_ENABLE_DXG_DETECTION=1
export TRITON_CACHE_AUTOTUNING=1
# Every reply runs on fresh threads; with glibc's default per-thread malloc arenas, freed audio/tensor buffers
# fragment across dozens of arenas and resident memory grows ~50 MB per conversation. Two arenas: ~1 MB.
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
# fla wraps a few helpers in torch.compile, which otherwise keeps 16 idle compile-worker processes (~0.6 GB)
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
exec .venv/bin/python server.py "$@"
