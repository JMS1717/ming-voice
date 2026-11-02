# Which AMD GPU to use, and the settings run.sh needs for it. Sourced by install.sh and the Docker entrypoint.

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

amd_env() {  # <index> <gfx> <GPU count>: export lines for .gpu_env (a note on stderr for an unsupported GPU)
  # an integrated GPU next to it: use only the big one
  [ "$3" -gt 1 ] && echo "export ROCR_VISIBLE_DEVICES=$1"
  # GPUs that PyTorch's ROCm build doesn't list, but that run as their closest supported sibling
  case $2 in
    gfx900 | gfx906 | gfx908 | gfx90a | gfx942 | gfx950 | gfx1030 | gfx1100 | gfx1101 | gfx1102 | gfx1150 | gfx1151 | gfx1200 | gfx1201) ;;
    gfx103*) echo "export HSA_OVERRIDE_GFX_VERSION=10.3.0" ;;
    gfx11*) echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0" ;;
    gfx120*) echo "export HSA_OVERRIDE_GFX_VERSION=12.0.1" ;;
    *) echo "$2 isn't supported by PyTorch's ROCm build; it may not work." >&2 ;;
  esac
  return 0
}
