#!/usr/bin/env bash
# Sourced by the setup and launch scripts from the repo root.
case "${MING_BACKEND:-ming}" in
    ming)
        MING_VENV=.venv
        MING_REQUIREMENTS=requirements.txt
        ;;
    lfm3-audio)
        MING_VENV=.venv-lfm3
        MING_REQUIREMENTS=requirements-lfm3.txt
        ;;
    *)
        echo "MING_BACKEND must be ming or lfm3-audio" >&2
        return 2
        ;;
esac
