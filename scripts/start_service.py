import logging
import sys

import uvicorn

logger = logging.getLogger(__name__)

from app.langchain_server import app

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
