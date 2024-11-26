# PostgreSQL RAG
Testing vLLM + RAG system

To build Docker image, run the following:

```sh
docker build -t vllm:cuda11.8 .
```

To run Docker container with LLMs:

```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name llm_server \
  -p 8888:8888 \
  -e RUN_MODE=server \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="mistralai/Mistral-Small-Instruct-2409" \
  -e HUGGING_FACE_HUB_TOKEN=<your-hugging-face-token> \
  vllm:cuda11.8

```

To run Docker container with VLMs (Vision Language Models):

```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name vlm_server \
  -p 8888:8888 \
  -e RUN_MODE=server \
  -e MODEL_TYPE=VLM \
  -e VLM_MODEL_NAME="llava-hf/llava-1.5-7b-hf" \
  -e FIXED_IMAGE_URL="https://example.com/your-image.jpg" \
  vllm:cuda11.8

```

To run Docker container with Streamlit UI:
```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name llm_ui \
  -p 8888:8888 \
  -e RUN_MODE=ui \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="mistralai/Mistral-Small-Instruct-2409" \
  -e HUGGING_FACE_HUB_TOKEN=<your-hugging-face-token> \
  vllm:cuda11.8

```

<!-- TODO: remove '| jq' from these curl commands -->
To query the FastAPI server:

```sh
curl -X POST "http://localhost:8888/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is the content of this image?\"}" | jq
```

To check FastAPI server health:

```sh
curl http://localhost:8888/health | jq
```

To query the Streamlit UI:

1. Since the container is running Streamlit on port `8888`, you need to set up SSH tunneling to access it from your local machine:
```sh
# Activate conda environment if not already active
conda activate llm  # or your preferred environment

# SSH tunneling
ssh -L 8888:localhost:8888 <your-username>@<machine>

```

2. Open your web browser and navigate to:
[http://localhost:8888](http://localhost:8888)
You should see the Streamlit UI where you can input queries and interact with the model.

## Database Permissions Setup

Before running tests, ensure proper permissions are set for the PostgreSQL data directory:

1. Add your user to the required group:
```bash
sudo usermod -aG kbradmin $USER   # Replace $USER with your username
```

2. Set correct permissions on the data directory:
```bash
sudo chmod -R 750 db_data
sudo chmod -R g+rx db_data
```

3. Verify permissions:
```bash
ls -ld db_data/  # Should show: drwxr-x--- owned by kbradmin:kbradmin
```

These steps ensure PostgreSQL can access its data directory while maintaining security. You may need to log out and back in for group changes to take effect.

Now you can run:
```bash
./test_db_data.sh --hf-token "your_token" --langchain-token "your_token"
```