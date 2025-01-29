FROM pytorch/pytorch:2.1.0-cpu

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

# Python dependecnies
RUN pip install --no-cache-dir --upgrade pip==24.0.0 && \
    pip install --no-cache-dir \
    cmake==3.28.1 \
    wheel==0.42.0 \
    packaging==23.2 \
    ninja==1.11.1 \
    setuptools-scm==8.0.0 \
    numpy==1.26.4

# Install other "external" python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install vLLM
RUN git clone --branch v${VLLM_VERSION} --depth 1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    # Patch _version.py to fix the syntax error
    sed -i "s/__version__ : str = version : str = '__version__ = '0.7.0'\\nversion = '0.7.0'/g" vllm/_version.py && \
    # Remove or replace the pinned torch requirement
    sed -i "s/torch==2.5.1+cpu/torch/g" requirements.txt setup.py && \
    # Build and install vLLM
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
