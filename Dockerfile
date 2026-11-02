# syntax=docker/dockerfile:1
# Ming Voice for a Linux host with an NVIDIA or AMD GPU (see "Docker" in README.md). The image holds the code and
# its Python env; the model (~20 GB, downloaded at the first start), your settings, keys, cloned voices and the
# compiled-kernel caches live in the /data volume.
#   docker build -t ming-voice:cuda .                       NVIDIA (CUDA 12.8; RTX 30 series or newer)
#   docker build --build-arg GPU=amd -t ming-voice:rocm .   AMD (ROCm 7.2)
FROM ubuntu:24.04
ARG GPU=nvidia
LABEL org.opencontainers.image.source=https://github.com/JMS1717/ming-voice \
      org.opencontainers.image.description="Ming Voice: a local, realtime speech-to-speech assistant" \
      org.opencontainers.image.licenses=MIT
# gcc: Triton compiles a small launcher at the first GPU kernel. openssl: the phone-access certificate.
# tini: passes `docker stop` on to the server.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates curl gcc libc6-dev openssl tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
# The Python env first: it changes far less often than the code
COPY setup_env.sh requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv MING_GPU=$GPU MING_SKIP_MODEL=1 bash setup_env.sh
COPY . .
ENV MING_GPU=$GPU MING_DATA_DIR=/data MING_MODEL_DIR=/data/model \
    MING_TURN_MODEL=/data/model_extra/smart-turn-v3.2-cpu.onnx \
    HF_HOME=/data/cache/huggingface XDG_CACHE_HOME=/data/cache TRITON_CACHE_DIR=/data/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor \
    MIOPEN_USER_DB_PATH=/data/cache/miopen MIOPEN_CUSTOM_CACHE_DIR=/data/cache/miopen \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000 8443 10300
HEALTHCHECK --start-period=20m --interval=1m \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/health" | grep -Eq '"ready": ?true'
ENTRYPOINT ["tini", "-g", "--", "docker/entrypoint.sh"]
