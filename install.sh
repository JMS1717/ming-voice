#!/usr/bin/env bash
# One-step installer for Ming Voice inside WSL2 Ubuntu 24.04 (AMD GPU; NVIDIA experimental). Safe to run again:
# done steps are skipped.
#   bash install.sh             ROCm on AMD (asks for your Linux password), Python env, model, Desktop shortcut
#   bash install.sh --no-rocm   skip the ROCm step (you installed it yourself)
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }
NO_ROCM=0; [ "${1:-}" = --no-rocm ] && NO_ROCM=1

# ---- checks
[ -e /dev/dxg ] || die "no GPU bridge (/dev/dxg). Run this inside WSL2 on Windows 11 with a recent AMD Adrenalin or NVIDIA driver."
. /etc/os-release
[ "${VERSION_ID:-}" = 24.04 ] || echo "Warning: Ming Voice is tested on Ubuntu 24.04; this is ${PRETTY_NAME:-unknown}."
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ -f model/model.safetensors.index.json ] || [ "$free_gb" -ge 30 ] || die "needs ~30 GB free disk here, found ${free_gb} GB."

# ---- basic tools
need=()
for c in git wget curl; do command -v "$c" >/dev/null || need+=("$c"); done
if [ ${#need[@]} -gt 0 ]; then
  say "Installing ${need[*]}"
  sudo apt-get update -qq && sudo apt-get install -y -qq "${need[@]}"
fi

# ---- ROCm + WSL GPU bridge
RESTART=0
gpu() { HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -oE 'gfx1[0-9a-f]+' || true; }
nv() { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true; }
if [ -z "${MING_GPU:-}" ] && [ -n "$(nv)" ]; then
  export MING_GPU=nvidia
  say "NVIDIA GPU found ($(nv)): using CUDA PyTorch, no ROCm needed. NVIDIA support is new and less tested than AMD."
elif [ "${MING_GPU:-}" = nvidia ]; then
  say "Skipping ROCm (MING_GPU=nvidia)"
elif [ -n "$(gpu)" ]; then
  say "ROCm is ready ($(gpu))"
elif [ $NO_ROCM = 1 ]; then
  say "Skipping ROCm (--no-rocm)"
else
  say "Installing ROCm 7.2.4 for WSL (about 10 minutes; asks for your Linux password)"
  sudo bash scripts/install_rocm_wsl.sh
  RESTART=1
fi

# ---- uv (Python env manager)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- Python env + models
say "Setting up Python and downloading the model (~20 GB, the first time only)"
bash setup_env.sh

# ---- Windows Desktop shortcut
if command -v powershell.exe >/dev/null && command -v wslpath >/dev/null; then
  target=$(wslpath -w "$PWD/windows/Ming Voice.cmd")
  if MING_LNK_TARGET="$target" WSLENV=MING_LNK_TARGET powershell.exe -NoProfile -Command \
    '$d = [Environment]::GetFolderPath("Desktop"); $s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d "Ming Voice.lnk")); $s.TargetPath = $env:MING_LNK_TARGET; $s.Description = "Start Ming Voice"; $s.Save()' >/dev/null 2>&1; then
    say "Added a 'Ming Voice' shortcut to your Windows Desktop"
  fi
fi

printf '\n\033[1;32mMing Voice is installed.\033[0m\n'
if [ $RESTART = 1 ]; then
  touch .wsl_restart_needed  # (install.ps1 restarts WSL for you when it sees this)
  [ -n "${MING_FROM_WINDOWS:-}" ] ||
    echo "One last step: restart WSL so the GPU permissions apply. In Windows, open PowerShell and run:  wsl --shutdown"
fi
echo "Then double-click 'Ming Voice' on your Desktop (or run ./run.sh here and open http://localhost:8000)."
