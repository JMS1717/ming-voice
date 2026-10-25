#!/usr/bin/env bash
# Build the Python env for Ming Voice and download the model. install.sh runs this for you.
# Needs: uv (https://docs.astral.sh/uv/; it brings Python 3.12), and the GPU driver:
#   NVIDIA (Linux or WSL)  the NVIDIA driver; PyTorch brings CUDA 12.8
#   AMD on Linux           the amdgpu kernel driver (any recent distro kernel); PyTorch brings ROCm 7.2
#   AMD in WSL             ROCm 7.2.4 for WSL (scripts/install_rocm_wsl.sh)
# The GPU is detected (an NVIDIA one when nvidia-smi lists it, AMD otherwise); force it with MING_GPU=amd|nvidia.
# MING_SKIP_MODEL=1 builds the env without downloading the model.
set -euo pipefail
cd "$(dirname "$0")"
GPU=${MING_GPU:-}
[ -n "$GPU" ] || { nvidia-smi -L 2>/dev/null | grep -q GPU && GPU=nvidia || GPU=amd; }
WSL=0; [ -e /dev/dxg ] && WSL=1
# uv's own Python build: Triton compiles a small C launcher against Python.h at the first kernel, and a distro
# Python often comes without its headers (python3-dev). An env made without them is made again.
pyinc() { .venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null; }
if [ -d .venv ] && [ ! -f "$(pyinc)/Python.h" ]; then
  echo "Rebuilding .venv: its Python has no C headers (Triton needs them)"
  rm -rf .venv
fi
[ -d .venv ] || uv venv -q -p 3.12 --managed-python .venv
. .venv/bin/activate
TORCH=(torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0)
if [ "$GPU" = nvidia ]; then
  # CUDA 12.8 (RTX 30 series or newer: bf16)
  CUDA_TORCH=("${TORCH[@]}" --index-url https://download.pytorch.org/whl/cu128)
  uv pip install "${CUDA_TORCH[@]}"
  uv pip install -r requirements.txt
  uv pip install "${CUDA_TORCH[@]}"  # (no-op unless a dependency swapped them)
elif [ $WSL = 0 ]; then
  # Native Linux: PyTorch's ROCm 7.2 build carries its own ROCm libraries, so no system ROCm is needed.
  ROCM=https://download.pytorch.org/whl/rocm7.2
  uv pip install "${TORCH[@]}" --index-url $ROCM
  uv pip install -r requirements.txt
  uv pip install "${TORCH[@]}" --index-url $ROCM  # (no-op unless a dependency swapped them)
  # openai-whisper pulls the CUDA `triton` from PyPI, which shadows the ROCm one (same module name).
  uv pip uninstall -q triton 2>/dev/null || true
  uv pip install --reinstall --no-deps triton-rocm==3.6.0 --index-url $ROCM
else
  # WSL: AMD's ROCm 7.2.4 build, which uses the system ROCm with its WSL (DXG) support.
  B=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4
  WHEELS=(
    "torch-2.11.0+rocm7.2.4.lw.git5fbd98f3-cp312-cp312-linux_x86_64.whl"
    "torchaudio-2.11.0+rocm7.2.4.git143129b5-cp312-cp312-linux_x86_64.whl"
    "torchvision-0.26.0+rocm7.2.4.git3d50b215-cp312-cp312-linux_x86_64.whl"
    "triton-3.7.0+rocm7.2.4.gitb4e20bbe-cp312-cp312-linux_x86_64.whl")
  mkdir -p wheels
  for f in "${WHEELS[@]}"; do
    [ -s "wheels/$f" ] || curl -fL --retry 3 -o "wheels/$f" "$B/${f//+/%2B}"
  done
  uv pip install "${WHEELS[@]/#/wheels/}"
  uv pip install -r requirements.txt
  # The pip triton/torch must stay the ROCm builds (other deps may try to pull CUDA ones).
  uv pip install --reinstall --no-deps "${WHEELS[@]/#/wheels/}"
  # Use the system (DXG-aware) HSA runtime instead of the bundled one.
  rm -f .venv/lib/python3.12/site-packages/torch/lib/libhsa-runtime64.so*
fi
python -c "import torch; print('PyTorch', torch.__version__)"
# ...and a C compiler. Without gcc or clang (immutable distros like Bazzite), use the clang that comes with Zig (PyPI).
if [ -z "${CC:-}" ] && ! command -v gcc >/dev/null && ! command -v clang >/dev/null; then
  echo "No C compiler found: installing Zig's into .venv for Triton"
  uv pip install -q ziglang
  printf '#!/bin/sh\nexec "%s" -m ziglang cc "$@"\n' "$PWD/.venv/bin/python" >.venv/bin/gcc
  chmod +x .venv/bin/gcc
fi

if [ "${MING_SKIP_MODEL:-0}" != 1 ]; then
  # Model (~20 GB) into ./model
  [ -f model/model.safetensors.index.json ] || hf download inclusionAI/Ming-3.0-audio-tiny --local-dir model
  # Smart Turn v3 end-of-turn classifier (8 MB, pipecat-ai, BSD-2) into ./model_extra
  mkdir -p model_extra
  [ -s model_extra/smart-turn-v3.2-cpu.onnx ] || hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir model_extra
  # Sample clip used by some dev tests (from the Ming-3.0-audio-flash repo, MIT)
  mkdir -p samples
  [ -s samples/introduce_ant_group.wav ] || curl -fsSL -o samples/introduce_ant_group.wav \
    https://raw.githubusercontent.com/inclusionAI/Ming-3.0-audio-flash/main/examples/audio/introduce_ant_group.wav || true
fi
echo "ENV_DONE ($GPU) - start with ./run.sh and open http://localhost:8000"
