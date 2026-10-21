#!/usr/bin/env bash
# Run the e2e test under several turn-taking configs: "spec_ms:smart_turn" pairs (default: new vs. old defaults)
cd "$(dirname "$0")/.."
cfgs=("$@")
[ ${#cfgs[@]} -eq 0 ] && cfgs=(32:1 64:1)
for cfg in "${cfgs[@]}"; do
  spec=${cfg%%:*}; smart=${cfg##*:}
  echo "== spec_ms $spec, smart turn $smart"
  E2E_SPEC_MS=$spec E2E_SMART=$smart timeout 300 .venv/bin/python dev/e2e_test.py 2>&1 | grep -E "^PASS|^FAIL|^end of speech|^withdrawn|^perceived"
done
