#!/usr/bin/env bash
# Build the Python env for Ming Voice on ROCm (WSL2) and download the model.
# Needs: ROCm installed (scripts/install_rocm_wsl.sh), python3.12, uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")"
B=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4
mkdir -p wheels
for f in \
  "torch-2.8.0+rocm7.2.4.lw.git6bea3e0b-cp312-cp312-linux_x86_64.whl" \
  "torchaudio-2.8.0+rocm7.2.4.git6e1c7fe9-cp312-cp312-linux_x86_64.whl" \
  "torchvision-0.23.0+rocm7.2.4.git824e8c87-cp312-cp312-linux_x86_64.whl" \
  "triton-3.4.0+rocm7.2.4.git0cace8d2-cp312-cp312-linux_x86_64.whl"; do
  [ -s "wheels/$f" ] || wget -q -O "wheels/$f" "$B/${f//+/%2B}"
done
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
uv pip install wheels/*.whl
uv pip install -r requirements.txt
# The pip triton/torch must stay the ROCm builds (other deps may try to pull CUDA ones).
uv pip install --reinstall --no-deps wheels/*.whl
# WSL: use the system (DXG-aware) HSA runtime instead of the bundled one.
rm -f .venv/lib/python3.12/site-packages/torch/lib/libhsa-runtime64.so*

# Model (~20 GB) into ./model
if [ ! -f model/model.safetensors.index.json ]; then
  hf download inclusionAI/Ming-3.0-audio-tiny --local-dir model
fi
# Sample clip used by some dev tests (from the Ming-3.0-audio-flash repo, MIT)
mkdir -p samples
[ -s samples/introduce_ant_group.wav ] || wget -q -O samples/introduce_ant_group.wav \
  https://raw.githubusercontent.com/inclusionAI/Ming-3.0-audio-flash/main/examples/audio/introduce_ant_group.wav || true
echo "ENV_DONE - start with ./run.sh and open http://localhost:8000"
