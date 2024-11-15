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

# Ensure psycopg2 is built against the correct libraries
RUN pip uninstall -y psycopg2-binary psycopg2 && \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    CPPFLAGS="-I/usr/include" \
    pip install --no-binary :all: psycopg2-binary

# Add library path explicitly
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

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

# Verify libffi installation and rebuild psycopg2
RUN ldconfig && \
    ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 && \
    ldd /usr/lib/x86_64-linux-gnu/libffi.so.7

# Add verification scripts
COPY verify_libs.sh verify_env.sh /usr/local/bin/
RUN echo "Verifying scripts exist..." && \
    ls -la /usr/local/bin/verify_libs.sh && \
    ls -la /usr/local/bin/verify_env.sh && \
    chmod +x /usr/local/bin/verify_libs.sh && \
    chmod +x /usr/local/bin/verify_env.sh && \
    echo "Scripts are executable"

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

# Make start script executable
RUN chmod +x /app/scripts/start.sh

RUN echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH" >> /root/.bashrc

# Set the entrypoint to our start script
ENTRYPOINT ["/app/scripts/start.sh"]
