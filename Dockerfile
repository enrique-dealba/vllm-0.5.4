FROM python:3.12-slim

ARG VLLM_VERSION=0.7.0
ENV VLLM_VERSION=${VLLM_VERSION}
ENV VLLM_TARGET_DEVICE=cpu

# Create non-root user
RUN groupadd -r vllm && useradd -r -g vllm vllm

WORKDIR /vllm-${VLLM_VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-12 \
    g++-12 \
    libnuma-dev \
    cmake \
    wget \
    git \
    libtcmalloc-minimal4 \
    libopenblas-dev \
    libdnnl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-12 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Now set TCMalloc after it's installed
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD

# Copy requirements
COPY --chown=vllm:vllm requirements.txt .

# Upgrade pip & install some basic python packages
RUN pip install --no-cache-dir --upgrade pip==25.0.0 && \
    pip install --no-cache-dir \
    cmake==3.28.1 \
    wheel==0.42.0 \
    packaging==23.2 \
    ninja==1.11.1 \
    setuptools-scm==8.0.0 \
    numpy==1.26.4

# ------------------------------------------------------------------
# 1) Install a valid CPU torch wheel before building vLLM
#    Example: torch==2.3.1+cpu from the official PyTorch CPU index.
# ------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other "external" python requirements
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# 2) Clone the vLLM repo at the specified tag/branch
# 3) Patch the syntax error in _version.py
# 4) Remove or override the pinned torch==2.5.1+cpu
# ------------------------------------------------------------------
RUN git clone --branch v${VLLM_VERSION} --depth 1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    echo "Current directory:" && \
    pwd && \
    echo "Directory contents:" && \
    ls -la && \
    echo "Finding version files:" && \
    find . -name "_version.py" && \
    echo "Finding setup files:" && \
    find . -name "setup.py" && \
    echo "Finding requirement files:" && \
    find . -name "requirements.txt" && \
    echo "Creating version file..." && \
    mkdir -p vllm && \
    echo "__version__ = '${VLLM_VERSION}'" > vllm/_version.py && \
    echo "version = '${VLLM_VERSION}'" >> vllm/_version.py && \
    echo "Version file contents:" && \
    cat vllm/_version.py && \
    echo "Updating torch requirements..." && \
    if [ -f "requirements.txt" ]; then \
        sed -i "s/torch==2.5.1+cpu/torch/g" requirements.txt; \
    fi && \
    if [ -f "setup.py" ]; then \
        sed -i "s/torch==2.5.1+cpu/torch/g" setup.py; \
    fi && \
    echo "Verifying torch installation:" && \
    python -c "import torch; print('PyTorch version:', torch.__version__)" && \
    echo "Installing vLLM..." && \
    VLLM_TARGET_DEVICE=cpu VLLM_CPU_AVX512BF16=0 python setup.py install && \
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

# Verify installation
RUN python -c "import vllm; print('vLLM version:', vllm.__version__)"

# Add CPU-specific environment variables
ENV VLLM_CPU_KVCACHE_SPACE=40
ENV VLLM_CPU_OMP_THREADS_BIND=0-7

ENTRYPOINT ["./scripts/start.sh"]
