import importlib
import json
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from statistics import mean
from typing import Any, Dict, Type
from uuid import uuid4

import requests
from langchain.callbacks.tracers import LangChainTracer
from PIL import Image
from pydantic import BaseModel

from app.config import settings

tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)


def load_image(url: str = settings.FIXED_IMAGE_URL) -> Image.Image:
    try:
        response = requests.get(url, timeout=settings.IMAGE_FETCH_TIMEOUT)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Failed to load image from {url}: {e}")
        raise


def load_schema() -> Type[BaseModel]:
    module = importlib.import_module("app.schemas.llm_responses")
    schema_class = getattr(module, settings.LLM_RESPONSE_SCHEMA)
    return schema_class


def log_to_langsmith(
    chain_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    metadata: Dict[str, Any],
):
    unique_id = str(uuid4())
    tracer.on_chain_start(
        {"name": chain_name},
        inputs,
        run_id=unique_id,
        tags=["streamlit_ui"],
        metadata=metadata,
    )
    tracer.on_chain_end(
        outputs=outputs,
        run_id=unique_id,
    )


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


def parse_intents(input_data):
    if isinstance(input_data, str):  # If it's a file path
        with open(input_data) as file:
            data = json.load(file)
    elif isinstance(input_data, (list, dict)):  # If it's already loaded data
        data = input_data
    else:
        raise TypeError("Expected file path (str) or loaded JSON data (list/dict)")

    summary = defaultdict(
        lambda: {
            "total_intents": 0,
            "status_counts": defaultdict(int),
            "update_types": defaultdict(int),
            "update_reasons": defaultdict(int),
            "status_progression": [],
            "priority": defaultdict(int),
            "frame_type": defaultdict(int),
            "num_frames": defaultdict(int),
            "integration_time_s": defaultdict(int),
            "track_type": defaultdict(int),
            # "average_completion_time": [],
        }
    )

    for intent in data:
        target = intent["target"]["name"]
        catalog_id = intent["target"]["rso"]["catalogId"]
        key = f"{target} (Catalog ID: {catalog_id})"

        summary[key]["total_intents"] += 1
        summary[key]["status_counts"][intent["currentStatus"]] += 1

        # Process update list
        status_progression = []
        for update in intent["updateList"]:
            summary[key]["update_types"][update["updateType"]] += 1
            summary[key]["update_reasons"][update["updateReason"]] += 1
            status_progression.append((update["status"], update["createdAt"]))

        # Sort status progression by timestamp and store
        status_progression.sort(key=lambda x: x[1])
        summary[key]["status_progression"].append(
            [status for status, _ in status_progression]
        )

        # Calculate completion time if applicable
        if status_progression and status_progression[-1][0] == "COMPLETED":
            start_time = datetime.fromisoformat(
                intent["createdAt"].replace("Z", "+00:00")
            )
            end_time = datetime.fromisoformat(
                status_progression[-1][1].replace("Z", "+00:00")
            )
            completion_time = (end_time - start_time).total_seconds()
            # summary[key]["average_completion_time"].append(completion_time)

        summary[key]["priority"][intent["priority"]] += 1
        params = intent["intentObservationParameters"]
        summary[key]["frame_type"][params["frameType"]] += 1
        summary[key]["num_frames"][params["numFrames"]] += 1
        summary[key]["integration_time_s"][params["integrationTimeS"]] += 1
        summary[key]["track_type"][params["trackType"]] += 1

    # Calculate average completion time
    # for key in summary:
    #     if summary[key]["average_completion_time"]:
    #         summary[key]["average_completion_time"] = sum(
    #             summary[key]["average_completion_time"]
    #         ) / len(summary[key]["average_completion_time"])
    #     else:
    #         summary[key]["average_completion_time"] = None

    return summary


def format_summary_intents(summary):
    formatted_summary = {}
    for key, data in summary.items():
        formatted_summary[key] = {
            "total_intents": data["total_intents"],
            "status_counts": dict(data["status_counts"]),
            "update_types": dict(data["update_types"]),
            "update_reasons": dict(data["update_reasons"]),
            "most_common_status_progression": max(
                set(tuple(prog) for prog in data["status_progression"]),
                key=data["status_progression"].count,
            ),
            "priority": dict(data["priority"]),
            "frame_type": dict(data["frame_type"]),
            "num_frames": dict(data["num_frames"]),
            "integration_time_s": dict(data["integration_time_s"]),
            "track_type": dict(data["track_type"]),
            # "average_completion_time": f"{data['average_completion_time']:.2f} seconds"
            # if data["average_completion_time"]
            # else "N/A",
        }
    return formatted_summary


def parse_collect_requests(input_data):
    if isinstance(input_data, str):  # If it's a file path
        with open(input_data) as file:
            data = json.load(file)
    elif isinstance(input_data, (list, dict)):  # If it's already loaded data
        data = input_data
    else:
        raise TypeError("Expected file path (str) or loaded JSON data (list/dict)")

    summary = defaultdict(
        lambda: {
            "total_requests": 0,
            "completed_requests": 0,
            "start_times": [],
            "end_times": [],
            "durations": [],
            "priority": defaultdict(int),
            "frame_type": defaultdict(int),
            "num_frames": defaultdict(int),
            "integration_time_s": defaultdict(int),
            "track_type": defaultdict(int),
            "sensor_names": set(),
            "sensor_locations": set(),
        }
    )

    for collect_request in data:
        target_name = collect_request["target"]["name"]
        catalog_id = collect_request["target"]["rso"]["catalogId"]
        key = f"{target_name} (Catalog ID: {catalog_id})"

        summary[key]["total_requests"] += 1

        if collect_request["intent"]["currentStatus"] == "COMPLETED":
            summary[key]["completed_requests"] += 1

        summary[key]["start_times"].append(collect_request["startDateTime"])
        summary[key]["end_times"].append(collect_request["endDateTime"])
        summary[key]["durations"].append(collect_request["durationS"])

        summary[key]["priority"][collect_request["priority"]] += 1
        summary[key]["frame_type"][collect_request["frameType"]] += 1

        intent_params = collect_request["intent"]["intentObservationParameters"]
        summary[key]["num_frames"][intent_params["numFrames"]] += 1
        summary[key]["integration_time_s"][intent_params["integrationTimeS"]] += 1
        summary[key]["track_type"][intent_params["trackType"]] += 1

        sensor = collect_request["instrument"]["sensor"]
        summary[key]["sensor_names"].add(sensor["name"])
        summary[key]["sensor_locations"].add(
            f"Lat: {sensor['latitudeDeg']}, Lon: {sensor['longitudeDeg']}, Alt: {sensor['altitudeKm']} km"
        )

    return summary


def format_summary_collects(summary, summary_type):
    formatted_summary = {}
    for key, data in summary.items():
        if summary_type == "intents":
            formatted_summary[key] = {
                "total_intents": data["total_intents"],
                "failed_count": data["failed_count"],
                "scheduled_count": data["scheduled_count"],
                "completed_count": data["completed_count"],
                "failure_reasons": list(data["failure_reasons"]),
                "priority": dict(data["priority"]),
                "frame_type": dict(data["frame_type"]),
                "num_frames": dict(data["num_frames"]),
                "integration_time_s": dict(data["integration_time_s"]),
                "track_type": dict(data["track_type"]),
            }
        elif summary_type == "collect_requests":
            formatted_summary[key] = {
                "total_requests": data["total_requests"],
                "completed_requests": data["completed_requests"],
                "completion_rate": f"{(data['completed_requests'] / data['total_requests']) * 100:.2f}%",
                "earliest_start": min(data["start_times"]),
                "latest_end": max(data["end_times"]),
                "avg_duration": f"{mean(data['durations']):.2f} seconds",
                "priority": dict(data["priority"]),
                "frame_type": dict(data["frame_type"]),
                "num_frames": dict(data["num_frames"]),
                "integration_time_s": dict(data["integration_time_s"]),
                "track_type": dict(data["track_type"]),
                "sensor_names": list(data["sensor_names"]),
                "sensor_locations": list(data["sensor_locations"]),
            }
    return formatted_summary
