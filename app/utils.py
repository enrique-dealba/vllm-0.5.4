import importlib
import json
import re
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
    """Load schema with caching that respects settings updates"""
    try:
        if (
            not hasattr(settings, "_schema_cache")
            or settings._schema_cache[0] != settings.LLM_RESPONSE_SCHEMA
        ):
            module = importlib.import_module("app.schemas.llm_responses")
            schema_class = getattr(module, settings.LLM_RESPONSE_SCHEMA)
            # Verify the schema class is properly defined
            if not issubclass(schema_class, BaseModel):
                raise TypeError(
                    f"Schema {settings.LLM_RESPONSE_SCHEMA} must be a Pydantic BaseModel"
                )
            settings._schema_cache = (settings.LLM_RESPONSE_SCHEMA, schema_class)
        return settings._schema_cache[1]
    except Exception as e:
        print(f"Error loading schema: {e}")
        raise


def debug_schema(schema: Type[BaseModel]) -> None:
    """Helper function to debug schema issues"""
    print("\nSchema Debug Info:")
    print(f"Schema name: {schema.__name__}")
    print(f"Schema fields: {schema.__fields__.keys()}")
    print(f"Schema base classes: {schema.__bases__}")
    if hasattr(schema, "model_fields"):
        print(f"Model fields: {schema.model_fields}")


def normalize_datetime_string(dt_str: str) -> str:
    """Extract datetime components in a consistent format."""
    # Extract components using regex
    # Match year, month, day, hour, minute, second, microsecond
    pattern = r"(\d{4})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{0,6})"
    match = re.search(pattern, dt_str)
    if match:
        year, month, day, hour, minute, second, micro = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
    return dt_str


def calculate_field_accuracy(fields):
    expected_fields = {
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 25,
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, 0, 250000, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, 0, 150000, tzinfo=TzInfo(UTC))",
        "orbital_regime": "LEO",
        "patience_minutes": 10,
        "priority": 12,
        "rso_id_list": [],
        "sensor_name_list": ["RME02", "LMNT01"],
    }

    correct_fields = 0
    total_fields = len(expected_fields)

    for field_name, expected_value in expected_fields.items():
        if field_name in fields:
            current_value = fields[field_name]

            # Handle datetime fields
            if field_name in ["objective_start_time", "objective_end_time"]:
                expected_normalized = normalize_datetime_string(str(expected_value))
                current_normalized = normalize_datetime_string(str(current_value))
                if expected_normalized == current_normalized:
                    correct_fields += 1
                continue

            # Handle lists (including empty lists)
            if isinstance(expected_value, list):
                try:
                    if sorted(str(x).strip() for x in current_value) == sorted(
                        str(x).strip() for x in expected_value
                    ):
                        correct_fields += 1
                except (TypeError, AttributeError):
                    pass
                continue

            # Regular field comparison
            if str(current_value).strip() == str(expected_value).strip():
                correct_fields += 1

    accuracy = (correct_fields / total_fields) * 100
    return accuracy, correct_fields, total_fields


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


FIELD_DISPLAY_CONFIG = {
    "response": {"title": "Response", "display_format": lambda x: x, "is_list": False},
    "sources": {
        "title": "Sources",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "evidence": {
        "title": "Supporting Evidence",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "confidence": {
        "title": "Confidence",
        "display_format": lambda x: f"{round(float(x) * 100, 1)}%",
        "is_list": False,
    },
    "classification_marking": {
        "title": "Classification",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "data_mode": {
        "title": "Data Mode",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "collect_request_type": {
        "title": "Collection Request Type",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "orbital_regime": {
        "title": "Orbital Regime",
        "display_format": lambda x: x,
        "is_list": False,
    },
    "patience_minutes": {
        "title": "Patience Time (Minutes)",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "end_time_offset_minutes": {
        "title": "End Time Offset (Minutes)",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "priority": {
        "title": "Priority Level",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
    "key_points": {
        "title": "Key Points",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "summary": {"title": "Summary", "display_format": lambda x: x, "is_list": False},
    "categories": {
        "title": "Categories",
        "display_format": lambda x: f"- {x}",
        "is_list": True,
    },
    "priority_level": {
        "title": "Priority Level",
        "display_format": lambda x: str(x),
        "is_list": False,
    },
}


def display_field(field_name: str, value: Any, writer_func=print) -> None:
    """Display a field based on its configuration.

    Args:
        field_name: Name of the field to display
        value: Value of the field
        writer_func: Function to use for output (default: print for testing)
    """
    if not value:  # Skip empty values
        return

    config = FIELD_DISPLAY_CONFIG.get(
        field_name,
        {
            "title": field_name.replace("_", " ").title(),
            "display_format": lambda x: x,
            "is_list": isinstance(value, (list, tuple)),
        },
    )

    writer_func(config["title"])  # Write header

    if config["is_list"]:
        for item in value:
            writer_func(config["display_format"](item))
    else:
        writer_func(config["display_format"](value))


def is_valid_field(field_name) -> bool:
    invalid_fields = {"response", "model_fields", "model_fields_set"}

    if (
        field_name.startswith("_")
        or field_name in invalid_fields
        or "model_" in field_name
    ):
        return False

    return True


def display_response(llm_response: Any, writer_func=print) -> None:
    """Display all available fields from the LLM response."""
    # Always display response field first if it exists
    if hasattr(llm_response, "response"):
        display_field("response", llm_response.response, writer_func)

    # Display all other fields
    for field_name in dir(llm_response):
        if is_valid_field(field_name):
            # writer_func(f"verbatim field_name: {field_name}")
            value = getattr(llm_response, field_name)
            if not callable(value):  # Skip methods
                display_field(field_name, value, writer_func)


def get_displayable_fields(llm_response: Any) -> Dict:
    displayable_fields = {}

    # Add response field first if it exists
    if hasattr(llm_response, "response"):
        displayable_fields["response"] = llm_response.response

    # Collect all other valid fields
    for field_name in dir(llm_response):
        if is_valid_field(field_name):
            value = getattr(llm_response, field_name)
            if not callable(value):  # Skip methods
                displayable_fields[field_name] = value

    return displayable_fields
