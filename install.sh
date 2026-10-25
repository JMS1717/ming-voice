#!/usr/bin/env bash
# One-step installer for Ming Voice, on Linux (AMD or NVIDIA) or in WSL2 Ubuntu on Windows. Safe to run again:
# finished steps are skipped.
#   curl -fsSL https://raw.githubusercontent.com/JMS1717/ming-voice/main/install.sh | bash
#   bash install.sh             from a clone
#   bash install.sh --no-rocm   WSL only: skip the ROCm step (you installed it yourself)
# Linux: Arch / CachyOS / EndeavourOS / Manjaro, Debian / Ubuntu / Pop!_OS / Mint, Fedora / Nobara, openSUSE.
# Settings: MING_DIR (where `curl | bash` puts it, default ~/ming-voice), MING_GPU=amd|nvidia (skip detection),
#           MING_SKIP_MODEL=1 (no model download), MING_REPO (git URL to install from, for forks).
set -euo pipefail
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
note() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }
NO_ROCM=0; for a in "$@"; do [ "$a" = --no-rocm ] && NO_ROCM=1; done
SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
WSL=0; [ -e /dev/dxg ] && WSL=1

# ---- system packages (git, curl, tar; openssl for phone access, PortAudio for the always-on client)
. /etc/os-release
FAMILY=
case " ${ID:-} ${ID_LIKE:-} " in
  *" arch "* | *" cachyos "* | *" manjaro "* | *" endeavouros "*) FAMILY=arch ;;
  *" debian "* | *" ubuntu "*) FAMILY=debian ;;
  *" fedora "* | *" rhel "* | *" centos "*) FAMILY=fedora ;;
  *suse*) FAMILY=suse ;;
esac
[ -e /run/ostree-booted ] && FAMILY=ostree  # Bazzite, Silverblue...: no package installs (they ship git and curl)

have_lib() { ls /usr/lib*/"$1"* /usr/lib/*-linux-gnu/"$1"* >/dev/null 2>&1; }
pkg() {  # generic name -> this distro's package
  case $FAMILY:$1 in
    debian:portaudio | suse:portaudio) echo libportaudio2 ;;
    *) echo "$1" ;;
  esac
}
install_pkgs() {  # (stdin from /dev/null: under `curl | bash` it is the rest of this script)
  case $FAMILY in
    # A missing or stale package list: sync it, and upgrade with it (Arch doesn't support partial upgrades).
    arch) $SUDO pacman -S --needed --noconfirm "$@" || $SUDO pacman -Syu --needed --noconfirm "$@" ;;
    debian) $SUDO apt-get update -qq && DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "$@" ;;
    fedora) $SUDO dnf install -y -q "$@" ;;
    suse) $SUDO zypper --non-interactive --quiet install --no-recommends "$@" ;;
    *) return 1 ;;
  esac
}
system_packages() {
  local need=() want=() c
  for c in git curl tar; do command -v "$c" >/dev/null || need+=("$(pkg "$c")"); done
  [ ${#need[@]} -eq 0 ] || [ "$FAMILY" != debian ] || need+=(ca-certificates)
  command -v openssl >/dev/null || want+=("$(pkg openssl)")
  [ $WSL = 1 ] || have_lib libportaudio.so.2 || want+=("$(pkg portaudio)")
  # Triton compiles a small launcher at the first GPU kernel (without one, setup_env.sh falls back to Zig's clang)
  if ! command -v gcc >/dev/null && ! command -v clang >/dev/null; then
    want+=(gcc); [ "$FAMILY" != debian ] || want+=(libc6-dev)
  fi
  [ $(( ${#need[@]} + ${#want[@]} )) -gt 0 ] || return 0
  say "Installing ${need[*]} ${want[*]}"
  if ! install_pkgs "${need[@]}" "${want[@]}" </dev/null; then
    [ ${#need[@]} -eq 0 ] || die "please install ${need[*]} with your package manager, then run this again."
    note "Couldn't install ${want[*]}: phone access needs openssl, the always-on mode needs PortAudio."
  fi
}
system_packages

# ---- run from `curl | bash`: fetch the repo, then run its own copy of this script
here=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)
if [ ! -f "$here/server.py" ]; then
  dir=${MING_DIR:-$HOME/ming-voice}
  if [ -d "$dir/.git" ]; then
    say "Updating $dir"
    git -C "$dir" pull --ff-only -q
  else
    say "Downloading Ming Voice into $dir"
    git clone -q --depth 1 "${MING_REPO:-https://github.com/JMS1717/ming-voice.git}" "$dir"
  fi
  exec bash "$dir/install.sh" "$@"
fi
cd "$here"

free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ -f model/model.safetensors.index.json ] || [ "${MING_SKIP_MODEL:-0}" = 1 ] || [ "$free_gb" -ge 30 ] || die "needs ~30 GB free disk here, found ${free_gb} GB."
[ "$(uname -m)" = x86_64 ] || die "needs an x86_64 PC (this is $(uname -m))."

# ---- GPU
nv() { nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 || true; }
nv_card() { grep -qs 0x10de /sys/bus/pci/devices/*/vendor && grep -qs '^0x03' /sys/bus/pci/devices/*/class; }
amd_best() {  # "<index among GPUs> <gfx name> <VRAM GiB> <GPU count>" of the AMD GPU with the most memory
  local n i=0 best="" bmem=0 v mem
  for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/* 2>/dev/null | sort -t/ -k8 -n); do
    v=$(awk '/gfx_target_version/ {print $2}' "$n/properties")
    [ "${v:-0}" -gt 0 ] || continue
    mem=$(cat "$n"/mem_banks/*/properties 2>/dev/null | awk '/size_in_bytes/ {s += $2} END {print int(s / 2^30)}')
    if [ "$mem" -gt "$bmem" ]; then
      bmem=$mem
      best="$i gfx$((v / 10000))$((v / 100 % 100))$(printf %x $((v % 100)))"
    fi
    i=$((i + 1))
  done
  [ -n "$best" ] && echo "$best $bmem $i"
}

RESTART=0 RELOGIN=0
if [ $WSL = 1 ]; then
  rocm() { HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -oE 'gfx1[0-9a-f]+' || true; }
  if [ -z "${MING_GPU:-}" ] && [ -n "$(nv)" ]; then
    export MING_GPU=nvidia
    say "NVIDIA GPU found ($(nv | cut -d, -f1-2) MiB): using CUDA PyTorch, no ROCm needed"
  elif [ "${MING_GPU:-}" = nvidia ]; then
    say "Skipping ROCm (MING_GPU=nvidia)"
  elif [ -n "$(rocm)" ]; then
    say "ROCm is ready ($(rocm))"
  elif [ $NO_ROCM = 1 ]; then
    say "Skipping ROCm (--no-rocm)"
  else
    say "Installing ROCm 7.2.4 for WSL (about 10 minutes; asks for your Linux password)"
    sudo bash scripts/install_rocm_wsl.sh
    RESTART=1
  fi
else
  amd=$(amd_best || true)
  if [ -z "${MING_GPU:-}" ]; then
    if [ -n "$(nv)" ]; then
      MING_GPU=nvidia
      # (an AMD GPU with more memory wins, e.g. a Radeon next to a small NVIDIA card)
      [ -n "$amd" ] && [ "$(echo "$amd" | cut -d' ' -f3)" -gt $(( $(nv | cut -d, -f2 | tr -dc 0-9) / 1024 )) ] && MING_GPU=amd
    elif [ -n "$amd" ]; then
      MING_GPU=amd
    elif nv_card; then
      die "found an NVIDIA GPU but no NVIDIA driver. Install your distro's NVIDIA driver (e.g. nvidia-open on Arch/CachyOS, akmod-nvidia on Fedora), reboot, and run this again."
    elif ls /dev/dri/renderD* >/dev/null 2>&1 && grep -qs 0x1002 /sys/class/drm/card*/device/vendor; then
      die "found an AMD GPU, but not as a compute device (/dev/kfd). Make sure the amdgpu driver is in use, then run this again."
    else
      die "no supported GPU found. Ming Voice needs an AMD (ROCm) or NVIDIA (CUDA) GPU with 16 GB+ of memory."
    fi
  fi
  export MING_GPU
  : >.gpu_env  # per-machine settings for run.sh
  if [ "$MING_GPU" = nvidia ] && [ -z "$(nv)" ]; then
    note "MING_GPU=nvidia, but nvidia-smi lists no GPU: installing anyway."
  elif [ "$MING_GPU" = nvidia ]; then
    IFS=, read -r name mem cc <<<"$(nv)"
    say "NVIDIA GPU: ${name# } ($(( ${mem// /} / 1024 )) GB)"
    [ "${cc// /}" \< 8.0 ] && note "This GPU is older than RTX 30 series (compute ${cc// /}); Ming Voice needs bf16, so it likely won't run."
    [ "$(( ${mem// /} / 1024 ))" -ge 15 ] || note "Ming Voice needs about 12.5 GB of GPU memory; this GPU may not have enough."
  elif [ -z "$amd" ]; then
    note "MING_GPU=amd, but no AMD compute device (/dev/kfd) is visible: installing anyway."
  else
    read -r idx gfx mem count <<<"$amd"
    say "AMD GPU: $gfx (${mem} GB)"
    [ "$mem" -ge 15 ] || note "Ming Voice needs about 12.5 GB of GPU memory; this GPU may not have enough."
    # an integrated GPU next to it: use only the big one
    [ "$count" -gt 1 ] && echo "export ROCR_VISIBLE_DEVICES=$idx" >>.gpu_env
    # GPUs that PyTorch's ROCm build doesn't list, but that run as their closest supported sibling
    case $gfx in
      gfx900 | gfx906 | gfx908 | gfx90a | gfx942 | gfx950 | gfx1030 | gfx1100 | gfx1101 | gfx1102 | gfx1150 | gfx1151 | gfx1200 | gfx1201) ;;
      gfx103*) echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" >>.gpu_env ;;
      gfx11*) echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" >>.gpu_env ;;
      gfx120*) echo "export HSA_OVERRIDE_GFX_VERSION=12.0.1" >>.gpu_env ;;
      *) note "$gfx isn't supported by PyTorch's ROCm build; it may not work." ;;
    esac
    if [ -e /dev/kfd ] && { ! [ -r /dev/kfd ] || ! [ -w /dev/kfd ]; }; then
      groups=$(for g in render video; do getent group "$g" >/dev/null && printf '%s,' "$g"; done)
      say "Giving $USER access to the GPU (groups: ${groups%,}; asks for your password)"
      $SUDO usermod -aG "${groups%,}" "$USER"
      RELOGIN=1
    fi
  fi
fi

# ---- uv (Python env manager; it also brings its own Python 3.12)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- Python env + models
say "Setting up Python and downloading the model (~20 GB, the first time only)"
bash setup_env.sh

# ---- shortcuts
if [ $WSL = 1 ]; then
  if command -v powershell.exe >/dev/null && command -v wslpath >/dev/null; then
    target=$(wslpath -w "$PWD/windows/Ming Voice.cmd")
    if MING_LNK_TARGET="$target" WSLENV=MING_LNK_TARGET powershell.exe -NoProfile -Command \
      '$d = [Environment]::GetFolderPath("Desktop"); $s = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d "Ming Voice.lnk")); $s.TargetPath = $env:MING_LNK_TARGET; $s.Description = "Start Ming Voice"; $s.Save()' >/dev/null 2>&1; then
      say "Added a 'Ming Voice' shortcut to your Windows Desktop"
    fi
  fi
else
  bash linux/ming-voice --install-shortcuts
fi

printf '\n\033[1;32mMing Voice is installed.\033[0m\n'
if [ $WSL = 1 ]; then
  if [ $RESTART = 1 ]; then
    touch .wsl_restart_needed  # (install.ps1 restarts WSL for you when it sees this)
    [ -n "${MING_FROM_WINDOWS:-}" ] ||
      echo "One last step: restart WSL so the GPU permissions apply. In Windows, open PowerShell and run:  wsl --shutdown"
  fi
  echo "Then double-click 'Ming Voice' on your Desktop (or run ./run.sh here and open http://localhost:8000)."
else
  [ $RELOGIN = 0 ] || note "One last step: log out and back in (or reboot) so the GPU access applies."
  echo "Start it from your app menu (Ming Voice), or run:  ming-voice"
  echo "  ming-voice --always-on   no browser: say \"Hey Ming, ...\" to this PC's mic any time"
  echo "  ming-voice --phone       use it from your phone too"
  echo "  ming-voice --help        everything else"
fi
