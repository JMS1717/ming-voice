#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -e /dev/dxg ] && export HSA_ENABLE_DXG_DETECTION=1  # WSL
# Per-machine GPU settings written by install.sh (which GPU to use, GFX version override)
[ -f .gpu_env ] && . ./.gpu_env
export TRITON_CACHE_AUTOTUNING=1
# Every reply runs on fresh threads; with glibc's default per-thread malloc arenas, freed audio/tensor buffers
# fragment across dozens of arenas and resident memory grows ~50 MB per conversation. Two arenas: ~1 MB.
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
# fla wraps a few helpers in torch.compile, which otherwise keeps 16 idle compile-worker processes (~0.6 GB)
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
[ "${MING_SUPERVISE:-0}" = 1 ] || exec .venv/bin/python server.py "$@"

# Always on (MING_SUPERVISE=1, as the Windows launcher starts it): start the server again when it crashes or hits
# a GPU error it can't recover from. Ctrl-C (exit 0) and "port taken, already running" (98) end it.
export MING_SUPERVISE
while :; do
    .venv/bin/python server.py "$@"
    rc=$?
    case $rc in 0 | 98 | 130) exit $rc ;; esac
    echo "$(date +%H:%M:%S) server exited with code $rc: restarting in 10 s (Ctrl-C to stop)"
    sleep 10 || exit 130
done
