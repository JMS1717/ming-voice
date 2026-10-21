#!/usr/bin/env bash
# Install ROCm 7.2.4 + the WSL GPU bridge (librocdxg) inside WSL2 Ubuntu 24.04.
# Prerequisite on Windows: a recent AMD Adrenalin driver with WSL support (RX 7900 / RX 9070 class GPU).
# Run with sudo:  sudo bash scripts/install_rocm_wsl.sh
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo "run with sudo" >&2; exit 1; fi
if [ ! -e /dev/dxg ]; then echo "/dev/dxg missing: this must run inside WSL2 with GPU support" >&2; exit 1; fi
export DEBIAN_FRONTEND=noninteractive
cd /tmp
wget -q https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
apt-get update
amdgpu-install -y --usecase=rocm --no-dkms
wget -q -O rocdxg.deb https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
dpkg -i rocdxg.deb
TARGET_USER="${SUDO_USER:-$USER}"
usermod -aG render,video "$TARGET_USER"
echo
echo "Done. Check with:  HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo | grep -E 'Marketing|gfx'"
