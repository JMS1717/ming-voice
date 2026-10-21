#!/usr/bin/env bash
# dev/test_encoder.py under math SDPA, then flash (AOTriton) SDPA; the second run compares with the first.
cd "$(dirname "$0")/.."
for f in 0 1; do
  echo "== TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=$f"
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=$f TRITON_CACHE_AUTOTUNING=1 timeout 900 .venv/bin/python dev/test_encoder.py 2>&1 \
    | grep -v -i "warn\|fasttext\|scaled_dot\|loaded in\|weights:"
done
