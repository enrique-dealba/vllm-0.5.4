FROM python:3.12.1-slim

ARG VLLM_VERSION=0.6.1
ENV VLLM_VERSION=${VLLM_VERSION}

# To build vLLM for CPU only
ENV VLLM_TARGET_DEVICE=cpu
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD

# Create non-root user `vllm`
RUN groupadd -r vllm && useradd -r -g vllm vllm

WORKDIR /vllm-${VLLM_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    gcc-12 \
    g++-12 \
    libnuma-dev \
    cmake \
    wget \
    git \
    libtcmalloc-minimal4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY --chown=vllm:vllm requirements.txt .

# Python dependecnies
RUN pip install --no-cache-dir --upgrade pip==24.0.0 && \
    pip install --no-cache-dir \
    cmake==3.28.1 \
    wheel==0.42.0 \
    packaging==23.2 \
    ninja==1.11.1 \
    setuptools-scm==8.0.0 \
    numpy==1.26.4 \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Build vLLM
RUN git clone --branch v${VLLM_VERSION} --depth 1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    python setup.py install && \
    cd .. && \
    rm -rf vllm

# Copy application files
COPY --chown=vllm:vllm app/ ./app/
COPY --chown=vllm:vllm scripts/ ./scripts/
COPY --chown=vllm:vllm tests/ ./tests/
COPY --chown=vllm:vllm Dockerfile ./Dockerfile

# Permissions
RUN chmod +x ./scripts/start.sh

# Switch to non-root `vllm` user
USER vllm

ENTRYPOINT ["./scripts/start.sh"]
