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

# vLLM Installation
# ------------------------------------------------------------------
# 2) Clone the vLLM repo at the specified tag/branch
# 3) Create a clean _version.py file with correct version information
# 4) Remove the torch requirement since we pre-installed it
# ------------------------------------------------------------------
# Enable shell debugging for these critical steps
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Clone and setup vLLM
RUN set -x && \
    git clone --branch v${VLLM_VERSION} https://github.com/vllm-project/vllm.git && \
    echo "Cloned vLLM repository."

WORKDIR /vllm-${VLLM_VERSION}/vllm

# Create a clean _version.py with correct syntax using a heredoc
RUN set -x && \
    mkdir -p vllm && \
    cat <<EOF > vllm/_version.py
__version__ = '${VLLM_VERSION}'
version = __version__
EOF && \
    echo "Created vllm/_version.py:" && \
    cat vllm/_version.py

# Remove torch== from setup.py and requirements.txt if they exist
RUN set -x && \
    if [ -f setup.py ]; then \
        sed -i '/torch==/d' setup.py && \
        echo "Removed 'torch==' from setup.py"; \
    else \
        echo "setup.py not found"; \
    fi && \
    if [ -f requirements.txt ]; then \
        sed -i '/torch==/d' requirements.txt && \
        echo "Removed 'torch==' from requirements.txt"; \
    else \
        echo "requirements.txt not found"; \
    fi && \
    # Verify that 'torch==' lines are removed
    echo "Verifying removal of 'torch==' lines:" && \
    if [ -f setup.py ]; then \
        grep 'torch==' setup.py || echo "'torch==' not found in setup.py"; \
    fi && \
    if [ -f requirements.txt ]; then \
        grep 'torch==' requirements.txt || echo "'torch==' not found in requirements.txt"; \
    fi

# Install vLLM
RUN set -x && \
    VLLM_TARGET_DEVICE=cpu VLLM_CPU_AVX512BF16=0 python setup.py install && \
    echo "Installed vLLM successfully."

WORKDIR /vllm-${VLLM_VERSION}

# Clean up cloned repository
RUN set -x && \
    rm -rf vllm && \
    echo "Removed cloned vllm repository."

# ------------------------------------------------------------------

# Copy application files
COPY --chown=vllm:vllm app/ ./app/
COPY --chown=vllm:vllm scripts/ ./scripts/
COPY --chown=vllm:vllm tests/ ./tests/
COPY --chown=vllm:vllm Dockerfile ./Dockerfile && \
    echo "Copied application files."

# Permissions
RUN chmod +x ./scripts/start.sh && \
    echo "Set execute permissions for start.sh."

# Switch to non-root vllm user
USER vllm && \
    echo "Switched to non-root user 'vllm'."

# Verify installation
RUN python -c "import vllm; print('vLLM version:', vllm.__version__)" && \
    echo "Verified vLLM installation."

# Add CPU-specific environment variables
ENV VLLM_CPU_KVCACHE_SPACE=40
ENV VLLM_CPU_OMP_THREADS_BIND=0-7 && \
    echo "Set CPU-specific environment variables."

ENTRYPOINT ["./scripts/start.sh"]
