#!/usr/bin/env bash
cd "$(dirname "$0")"
export HSA_ENABLE_DXG_DETECTION=1
export TRITON_CACHE_AUTOTUNING=1
exec .venv/bin/python server.py "$@"
