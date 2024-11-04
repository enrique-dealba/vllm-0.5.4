import logging
from typing import Dict

from app.model import ModelManager

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self, gpu_assignments: Dict[str, int]):
        self.managers = {}
        for service_name, gpu_id in gpu_assignments.items():
            self.managers[service_name] = ModelManager.get_instance(gpu_id)
            logger.info(f"Initialized ModelManager for {service_name} on GPU {gpu_id}")

    async def forward_request(self, service_name: str, request_data: Dict) -> Dict:
        manager = self.managers.get(service_name)
        if not manager:
            raise ValueError(f"Unknown service: {service_name}")

        llm = manager.get_llm()
        if not llm:
            raise RuntimeError(f"LLM not initialized for {service_name}")

        query = request_data.get("text")
        if not query:
            raise ValueError("No 'text' field in request data")

        response = await llm.agenerate([query])
        return {"response": response.generations[0][0].text}
