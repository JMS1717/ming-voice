#!/usr/bin/env bash
# Container start: download the models into /data the first time, then run the server under run.sh's supervisor,
# with phone access (MING_PHONE=1) and the Home Assistant bridge (MING_WYOMING=1) next to it.
set -euo pipefail
cd /app
. .venv/bin/activate
mkdir -p "$MING_DATA_DIR"

if [ "$MING_GPU" = nvidia ] && [ ! -e /dev/nvidiactl ]; then
  echo "No NVIDIA GPU in the container: start it with --gpus all (docker-compose.yml does), and install the"
  echo "NVIDIA Container Toolkit on the host."
  exit 1
fi
if [ "$MING_GPU" = amd ]; then
  if [ ! -e /dev/kfd ]; then
    echo "No AMD GPU in the container: start it with --device /dev/kfd --device /dev/dri (docker-compose.yml does)."
    exit 1
  fi
  # Which GPU, and the GFX override PyTorch may need (as install.sh does), unless you set them yourself
  if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}${ROCR_VISIBLE_DEVICES:-}${HIP_VISIBLE_DEVICES:-}" ]; then
    . scripts/amd_gpu.sh
    if amd=$(amd_best); then
      read -r idx gfx mem count <<<"$amd"
      echo "AMD GPU: $gfx ($mem GB)"
      amd_env "$idx" "$gfx" "$count" >.gpu_env
    fi
  fi
fi

if [ ! -f "$MING_MODEL_DIR/model.safetensors.index.json" ]; then
  echo "Downloading the model (~20 GB, the first start only) into $MING_MODEL_DIR"
  hf download inclusionAI/Ming-3.0-audio-tiny --local-dir "$MING_MODEL_DIR"
fi
# Smart Turn v3 end-of-turn classifier (8 MB)
[ -s "$MING_TURN_MODEL" ] ||
  hf download pipecat-ai/smart-turn-v3 "$(basename "$MING_TURN_MODEL")" --local-dir "$(dirname "$MING_TURN_MODEL")" >/dev/null

# (the phone proxy and the bridge find the server on the ports it was given)
export MING_LAN_TARGET=${MING_LAN_TARGET:-127.0.0.1:${MING_LAN_PORT:-8001}}
export MING_SERVER=${MING_SERVER:-http://127.0.0.1:${PORT:-8000}}
if [ "${MING_PHONE:-0}" = 1 ]; then
  python windows/lan_proxy.py &
  echo "Phone access: https on port ${MING_PHONE_PORT:-8443}. Press Phone in the app for the QR code."
fi
if [ "${MING_WYOMING:-0}" = 1 ]; then
  python wyoming_bridge.py &
  echo "Home Assistant bridge: Wyoming protocol on port ${MING_WYOMING_PORT:-10300}."
fi
export MING_SUPERVISE=1
set +e
./run.sh 2>&1 | grep --line-buffered -vE 'Warning|warn\(|bos_token|eos_token|scaled_dot|theta is None'
rc=${PIPESTATUS[0]}
if [ "$rc" = 3 ]; then
  # run.sh gave up (the model didn't load three times): stay up rather than let Docker's restart policy load it
  # again every minute
  echo "Fix what's above (e.g. close programs that hold VRAM), then: docker compose restart"
  exec sleep infinity
fi
exit "$rc"
