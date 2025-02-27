FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV VLLM_VERSION=0.6.1
ENV PYTHON_VERSION=310
ENV PYTORCH_MULTIPROCESSING_START_METHOD=spawn

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh

# Add conda to path
ENV PATH="/root/miniconda3/bin:${PATH}"

# Create and activate conda environment
RUN conda create -n vllm python=3.10 -y
SHELL ["conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Install vLLM with CUDA 11.8
RUN pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install ALL pip dependencies
RUN pip install -r requirements.txt && \
    pip install pytest pytest-cov

# Copy rest of application files
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Make start script executable
RUN chmod +x /app/scripts/start.sh

# Set the entrypoint to our start script
ENTRYPOINT ["/app/scripts/start.sh"]
