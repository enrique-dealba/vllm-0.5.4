# Dockerfile (updated)
FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV VLLM_VERSION=0.6.1
ENV PYTHON_VERSION=310
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/app

# Install system dependencies and proper library versions
RUN apt-get update && apt-get install -y \
    wget \
    libpq-dev \
    python3-dev \
    gcc \
    && wget http://archive.ubuntu.com/ubuntu/pool/main/p/p11-kit/libp11-kit0_0.23.20-1ubuntu0.1_amd64.deb \
    && wget http://archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb \
    && wget http://archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi-dev_3.3-4_amd64.deb \
    && dpkg -i libp11-kit0_0.23.20-1ubuntu0.1_amd64.deb \
    && dpkg -i libffi7_3.3-4_amd64.deb \
    && dpkg -i libffi-dev_3.3-4_amd64.deb \
    && rm *.deb \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and set up environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

# Create and activate conda environment
RUN conda create -n vllm python=3.10 -y
SHELL ["conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Install vLLM with CUDA support
RUN pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install psycopg2 with proper flags
RUN LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    CPPFLAGS="-I/usr/include" \
    pip install --no-binary :all: psycopg2-binary==2.9.9

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY init-db.sh ./init-db.sh

# Make scripts executable
RUN chmod +x /app/scripts/start.sh \
    && chmod +x /app/scripts/verify_*.sh \
    && chmod +x /app/init-db.sh

# Install project dependencies
RUN pip install -r requirements.txt \
    && pip install langchain langchain_community -q

# Setup library path
RUN echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH" >> /root/.bashrc

# Verify environment and install order
RUN ldconfig && \
    ldd $(find /usr/lib/x86_64-linux-gnu -name "libffi.so*") && \
    python3 -c "import psycopg2; print('psycopg2 imported successfully')"

ENTRYPOINT ["/app/scripts/start.sh"]