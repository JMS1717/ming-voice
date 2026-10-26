#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -e /dev/dxg ] && export HSA_ENABLE_DXG_DETECTION=1  # WSL
# Per-machine GPU settings written by install.sh (which GPU to use, GFX version override)
[ -f .gpu_env ] && . ./.gpu_env
export TRITON_CACHE_AUTOTUNING=1
# The C compiler setup_env.sh put in .venv when the system has none (Triton builds its launcher with it)
[ -z "${CC:-}" ] && [ -x .venv/bin/gcc ] && export CC="$PWD/.venv/bin/gcc"
# Every reply runs on fresh threads; with glibc's default per-thread malloc arenas, freed audio/tensor buffers
# fragment across dozens of arenas and resident memory grows ~50 MB per conversation. Two arenas: ~1 MB.
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
# fla wraps a few helpers in torch.compile, which otherwise keeps 16 idle compile-worker processes (~0.6 GB)
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
[ "${MING_SUPERVISE:-0}" = 1 ] || exec .venv/bin/python server.py "$@"

# Always on (MING_SUPERVISE=1, as the Windows launcher starts it): start the server again when it crashes or hits
# a GPU error it can't recover from. Ctrl-C (exit 0) and "port taken, already running" (98) end it, and so do
# three failed starts in a row (3: the model didn't load, e.g. other programs hold too much VRAM).
export MING_SUPERVISE
failed=0
while :; do
    .venv/bin/python server.py "$@"
    rc=$?
    case $rc in
    0 | 98 | 130) exit $rc ;;
    76) export MING_INT8_EXPERTS=1 ;; # out of VRAM while loading (other programs hold some): int8 experts fit
    esac
    if [ $rc = 3 ]; then failed=$((failed + 1)); else failed=0; fi
    if [ $failed -ge 3 ]; then
        echo "$(date +%H:%M:%S) the server failed to start 3 times in a row: see the errors above"
        exit $rc
    fi
    echo "$(date +%H:%M:%S) server exited with code $rc: restarting in 10 s (Ctrl-C to stop)"
    sleep 10 || exit 130
done
