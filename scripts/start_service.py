import os
import sys

# Set CUDA device before any other imports
cuda_device = os.environ.get("CUDA_DEVICE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(f"Setting CUDA_VISIBLE_DEVICES={cuda_device}")

import uvicorn

from app.langchain_server import app, verify_gpu_setup

if __name__ == "__main__":
    # Verify GPU setup before starting
    if not verify_gpu_setup():
        print("GPU setup verification failed!")
        sys.exit(1)

    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
