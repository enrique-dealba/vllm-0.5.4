# vllm-0.5.4
Testing vLLM 0.5.4

To build Docker image:

```sh
docker build -t vllm:cuda11.8 .
```

To run Docker container:

```sh
docker run -v ~/.cache/huggingface:/root/.cache/huggingface --gpus all --name vlm -p 8888:8888 vllm:0.5.4-cuda11.8
```

To query the FastAPI server:

```sh
curl -X POST "http://localhost:8888/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is the content of this image?\"}" | jq
```
