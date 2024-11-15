FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV VLLM_VERSION=0.6.1
ENV PYTHON_VERSION=310
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH

# Install specific versions of required libraries
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

# Install psycopg2 in conda environment
RUN LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    CPPFLAGS="-I/usr/include" \
    pip install --no-binary :all: psycopg2-binary==2.9.9

# Verify libraries
RUN ldconfig && \
    ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 && \
    ldd /usr/lib/x86_64-linux-gnu/libffi.so.7

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install project dependencies
RUN pip install -r requirements.txt

# Install LangChain and LangChain Community
RUN pip install langchain langchain_community -q

# Copy verification scripts
COPY scripts/verify_libs.sh scripts/verify_env.sh /usr/local/bin/

# Make scripts executable
RUN chmod +x /usr/local/bin/verify_libs.sh /usr/local/bin/verify_env.sh

# Create verification scripts
RUN echo '#!/bin/bash\n\
source /root/miniconda3/bin/activate vllm\n\
set -x\n\
echo "=== Library Verification Start ==="\n\
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"\n\
ldconfig -p | grep libffi\n\
ldconfig -p | grep p11-kit\n\
ldd /usr/lib/x86_64-linux-gnu/libffi.so.7\n\
ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0\n\
python3 -c "import psycopg2; print(\"psycopg2 version:\", psycopg2.__version__)"' > /usr/local/bin/verify_libs.sh && \
    echo '#!/bin/bash\n\
source /root/miniconda3/bin/activate vllm\n\
set -x\n\
echo "=== Environment Verification Start ==="\n\
echo "PYTHONPATH: $PYTHONPATH"\n\
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"\n\
echo "Current working directory: $(pwd)"\n\
echo "Python executable: $(which python3)"\n\
python3 --version' > /usr/local/bin/verify_env.sh

# Make start script executable
RUN chmod +x /app/scripts/start.sh

# Setup library path in bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH" >> /root/.bashrc

# Set the entrypoint to our start script
ENTRYPOINT ["/app/scripts/start.sh"]