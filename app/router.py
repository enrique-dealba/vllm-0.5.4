import logging
from typing import Dict

from app.model import ModelManager

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, service_gpu_map: Dict[str, int]):
        self.managers = {}
        for service_name, gpu_id in service_gpu_map.items():
            try:
                self.managers[service_name] = ModelManager.get_instance(gpu_id)
                logger.info(
                    f"Initialized ModelManager for {service_name} on GPU {gpu_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize {service_name} on GPU {gpu_id}: {e}"
                )
                raise

    async def forward_request(self, service_name: str, request_data: Dict):
        if service_name not in self.managers:
            raise ValueError(f"Unknown service: {service_name}")

        manager = self.managers[service_name]
        llm = manager.get_llm()

        if not llm:
            raise ValueError(f"LLM not initialized for service: {service_name}")

        try:
            response = llm.invoke(request_data["text"])
            return {"response": response}
        except Exception as e:
            logger.error(f"Error processing request for {service_name}: {e}")
            raise
