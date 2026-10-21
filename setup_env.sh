#!/usr/bin/env bash
# Build the Python env for Ming Voice (WSL2) and download the model.
# Needs: python3.12, uv (https://docs.astral.sh/uv/), and for AMD: ROCm installed (scripts/install_rocm_wsl.sh).
# The GPU is detected (an NVIDIA one when nvidia-smi lists it, AMD otherwise); force it with MING_GPU=amd|nvidia.
set -euo pipefail
cd "$(dirname "$0")"
GPU=${MING_GPU:-}
[ -n "$GPU" ] || { nvidia-smi -L 2>/dev/null | grep -q GPU && GPU=nvidia || GPU=amd; }
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
if [ "$GPU" = nvidia ]; then
  # Same PyTorch / Triton versions as the ROCm build, for CUDA 12.8 (RTX 30 series or newer: bf16).
  CUDA_TORCH=(torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128)
  uv pip install "${CUDA_TORCH[@]}"
  uv pip install -r requirements.txt
  uv pip install "${CUDA_TORCH[@]}"  # (no-op unless a dependency swapped them)
else
  B=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4
  mkdir -p wheels
  for f in \
    "torch-2.8.0+rocm7.2.4.lw.git6bea3e0b-cp312-cp312-linux_x86_64.whl" \
    "torchaudio-2.8.0+rocm7.2.4.git6e1c7fe9-cp312-cp312-linux_x86_64.whl" \
    "torchvision-0.23.0+rocm7.2.4.git824e8c87-cp312-cp312-linux_x86_64.whl" \
    "triton-3.4.0+rocm7.2.4.git0cace8d2-cp312-cp312-linux_x86_64.whl"; do
    [ -s "wheels/$f" ] || wget -q -O "wheels/$f" "$B/${f//+/%2B}"
  done
  uv pip install wheels/*.whl
  uv pip install -r requirements.txt
  # The pip triton/torch must stay the ROCm builds (other deps may try to pull CUDA ones).
  uv pip install --reinstall --no-deps wheels/*.whl
  # WSL: use the system (DXG-aware) HSA runtime instead of the bundled one.
  rm -f .venv/lib/python3.12/site-packages/torch/lib/libhsa-runtime64.so*
fi

# Model (~20 GB) into ./model
if [ ! -f model/model.safetensors.index.json ]; then
  hf download inclusionAI/Ming-3.0-audio-tiny --local-dir model
fi
# Smart Turn v3 end-of-turn classifier (8 MB, pipecat-ai, BSD-2) into ./model_extra
mkdir -p model_extra
[ -s model_extra/smart-turn-v3.2-cpu.onnx ] || hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir model_extra
# Sample clip used by some dev tests (from the Ming-3.0-audio-flash repo, MIT)
mkdir -p samples
[ -s samples/introduce_ant_group.wav ] || wget -q -O samples/introduce_ant_group.wav \
  https://raw.githubusercontent.com/inclusionAI/Ming-3.0-audio-flash/main/examples/audio/introduce_ant_group.wav || true
echo "ENV_DONE ($GPU) - start with ./run.sh and open http://localhost:8000"
