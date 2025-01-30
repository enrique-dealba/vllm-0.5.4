FROM semmtech/nlp-base:torch-2.5.1-cpu-20241120

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
# RUN pip install --no-cache-dir \
#     torch==2.3.1+cpu \
#     -f https://download.pytorch.org/whl/cpu/torch_stable.html

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

WORKDIR /vllm-${VLLM_VERSION}

# Install numpy first to avoid warnings
RUN pip install --no-cache-dir numpy==1.26.4

# Create version file
RUN set -x && \
    mkdir -p app && \
    echo "# Version information" > app/_version.py && \
    echo "__version__ = '${VLLM_VERSION}'" >> app/_version.py && \
    echo "version = __version__" >> app/_version.py

# Handle vLLM installation with extensive CPU configuration
RUN set -x && \
    cd vllm && \
    # Force CPU-only configuration
    export USE_CUDA=0 && \
    export CUDA_VISIBLE_DEVICES="" && \
    export VLLM_TARGET_DEVICE=cpu && \
    export VLLM_CPU_AVX512BF16=0 && \
    export TORCH_CUDA_ARCH_LIST="" && \
    export VLLM_PYTHON_EXECUTABLE=$(which python3) && \
    # Modify CMake configuration
    sed -i 's/find_package(Torch REQUIRED)/find_package(Torch REQUIRED CPU)/g' CMakeLists.txt && \
    sed -i 's/if(CUDA_FOUND)/if(FALSE)/g' CMakeLists.txt && \
    # Install package
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
                -DVLLM_TARGET_DEVICE=cpu \
                -DUSE_CUDA=OFF \
                -DTORCH_CUDA_ARCH_LIST= \
                -DVLLM_PYTHON_EXECUTABLE=$(which python3)" \
    VLLM_TARGET_DEVICE=cpu \
    VLLM_CPU_AVX512BF16=0 \
    pip install . \
        --no-cache-dir \
        --verbose && \
    echo "Installed vLLM successfully."

WORKDIR /vllm-${VLLM_VERSION}

# Verify the installation
RUN python3 -c "import vllm; print('vLLM version:', vllm.__version__)"

# Clean up cloned repository
RUN set -x && \
    rm -rf vllm && \
    echo "Removed cloned vllm repository."

# Copy application files
COPY --chown=vllm:vllm app/ ./app/
COPY --chown=vllm:vllm scripts/ ./scripts/
COPY --chown=vllm:vllm tests/ ./tests/
COPY --chown=vllm:vllm Dockerfile ./Dockerfile

# Permissions
RUN chmod +x ./scripts/start.sh

# Switch to non-root vllm user
USER vllm

# Verify installation
RUN python -c "import vllm; print('vLLM version:', vllm.__version__)"

# Add CPU-specific environment variables
ENV VLLM_CPU_KVCACHE_SPACE=40
ENV VLLM_CPU_OMP_THREADS_BIND=0-7

ENTRYPOINT ["./scripts/start.sh"]
