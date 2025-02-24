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

import pytz
import requests
from langchain.callbacks.tracers import LangChainTracer
from PIL import Image
from pydantic import BaseModel

from app.config import settings

tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)


OBJECTIVE_TEST_CASES = {
    # Example 1: CatalogMaintenanceObjective
    (
        "Create a CatalogMaintenanceObjective for sensors RME04 and LMNT02 with U markings, "
        "TEST mode, priority 12, patience of 10 mins, end time offset of 25 mins, visibility check false. "
        "Start at 2024-05-21 19:20:00+00:00, end at 2024-05-21 22:30:00+00:00. Use RATE_TRACK_SIDEREAL tracking in LEO regime. "
        "RSO ID list includes '12445,67889'."
    ): {
        "binning": None,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 25,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2024, 5, 21, 22, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2024, 5, 21, 19, 20, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "LEO",
        "patience_minutes": 10,
        "priority": 12,
        "rso_id_list": ["12445", "67889"],
        "sensor_name_list": ["RME04", "LMNT02"],
        "visibility_check": False,
    },
    # Example 2: SearchObjective
    (
        "Create a SearchObjective for target 12345 using sensor UKR12. Set S marking, REAL mode, priority 5, "
        "RATE_TRACK_SIDEREAL tracking. Start at 2024-07-24 09:30:00+00:00, end at 2024-07-24 11:00:00+00:00. "
        "Initial offset 60 seconds, final offset 90 seconds, frame overlap 70%, end time offset 45 minutes. "
        "Use search type ALONG_TRACK with search start time 15 minutes after objective start."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 45,
        "final_offset": 90,
        "frame_overlap_percentage": 0.7,
        "frame_type": "LIGHT",
        "initial_offset": 60,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2024, 7, 24, 11, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2024, 7, 24, 9, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 5,
        "search_start_time": "datetime.datetime(2024, 7, 24, 10, 45, tzinfo=TzInfo(UTC))",
        "search_type": "ALONG_TRACK",
        "sensor_name": "UKR12",
        "target_id": "12345",
        "visibility_check": False,
    },
    # Example 3: GeodssRevisitObjective
    (
        "Create a GeodssRevisitObjective for targets 12345,67890 using sensors RME15,LMNT17. Set U//FOUO marking, REAL mode, "
        "priority 10, RATE_TRACK_SIDEREAL tracking. Start at 2024-08-11 19:20:00+00:00. Set readout_rate 1 (2MHz), gain_setting 0 (High Gain), "
        "soi_filter 1 (1% Light), auto_track_type 1 (Automatic), camera_mode 0 (Normal), array_kind 0 (Main), binning_mode 1 (HW Binning), "
        "scan_mode 1 (Single Frame)."
    ): {
        "acquisition_type": 0,
        "array_kind": 0,
        "auto_track_roi_position": 0,
        "auto_track_type": 1,
        "binning_mode": 1,
        "camera_mode": 0,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 0,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,
        "num_skip_frames": 0,
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2024, 8, 11, 19, 20, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 30,
        "priority": 10,
        "rate_track_verify": 0,
        "readout_rate_setting": 1,
        "revisits_per_hour": None,
        "scan_mode": 1,
        "sensor_name_list": ["RME15", "LMNT17"],
        "soi_filter_position": 1,
        "target_id_list": ["12345", "67890"],
        "visibility_check": False,
    },
    # Example 4: PeriodicRevisitObjective
    (
        "Create a PeriodicRevisitObjective for targets 12225,68887 using sensors RME05,LMNT06. "
        "Set S marking, TEST mode, priority 2, patience minutes 30, ignore other objective submissions false. "
        "Start objective at 2024-06-21 19:20:00+00:00. Set optimal frames per hour 400, number of frames 5, integration time 2 seconds."
    ): {
        "classification_marking": "S",
        "target_id_list": ["12225", "68887"],
        "sensor_name_list": ["RME05", "LMNT06"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": None,
        "number_of_frames": 5,
        "integration_time": 2,
        "binning": None,
        "objective_start_time": "datetime.datetime(2024, 6, 21, 19, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 2,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # Example 5: UctObservationObjective
    (
        "Create a UctObservationObjective for UCT RSOs 12345,67890 using sensors RME18, LMNT19. "
        "Set U//FOUO marking, REAL mode, GEO regime, priority 10, 6 revisits per hour. "
        "Start at 2024-09-12 19:20:00+00:00. Enable sorting by brightest UCT, set end time offset to 60 minutes, visibility check true. "
        "Set number of frames to 5 and integration time 2 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["12345", "67890"],
        "sensor_name_list": ["RME18", "LMNT19"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 6.0,
        "number_of_frames": 5,
        "integration_time": 2,
        "binning": None,
        "end_time_offset_minutes": 60,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2024, 9, 12, 19, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 10,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # Example 6: SingleIntentObjective
    (
        "Create a SingleIntentObjective with target ID 11223, RSO ID 66778, using sensors RME22,LMNT24. "
        "Set U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, priority 10. "
        "Start objective at 2024-10-01 11:20:00+00:00. Set number of frames to 5, integration time 2 seconds, binning 2."
    ): {
        "classification_marking": "U",
        "target_id": "11223",
        "rso_id": "66778",
        "sensor_name_list": ["RME22", "LMNT24"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 5,
        "integration_time": 2,
        "priority": 10,
        "binning": 2,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2024, 10, 1, 11, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # Example 7: DataEnrichmentObjective
    (
        "Create a DataEnrichmentObjective for targets 55441, 99886, 50051 using sensors RME31,LMNT34. "
        "Set U//FOUO marking, REAL mode, RATE_TRACK tracking, and set max RSO to observe as 8, 10 revisits per hour. "
        "Start at 2025-01-21 08:00:00+00:00. Set visibility check true."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["55441", "99886", "50051"],
        "sensor_name_list": ["RME31", "LMNT34"],
        "collect_request_type": "RATE_TRACK",  # overridden to RATE_TRACK
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 8,
        "revisits_per_hour": 10,
        "objective_start_time": "datetime.datetime(2025, 1, 21, 8, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # Example 8: SensorCheckoutObjective
    (
        "Create a SensorCheckoutObjective with classification_marking='U' and sensor_name='RME01'. "
        "Set data_mode='REAL', orbital_regime='GEO', collect_request_type='RATE_TRACK_SIDEREAL', priority=10, "
        "revisits_per_hour=1.0, objective_start_time='2025-02-01 19:20:00+00:00', visibility_check=true, "
        "patience_minutes=30, number_of_frames=5, integration_time=2."
    ): {
        "classification_marking": "U",
        "sensor_name": "RME01",
        "orbital_regime": "GEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 1.0,
        "number_of_frames": 5,
        "integration_time": 2,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 2, 1, 19, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 10,
        "objective_name": "SensorCheckoutObjective",
    },
    # Example 9: BaselineAutonomyObjective
    (
        "Create a BaselineAutonomyObjective with UUID '123e4567-e89b-12d3-a456-426614174000'. "
        "Use U markings, REAL mode, LIGHT frame type, priority 1000. "
        "Use following RSO ids: 11112, 99996, and 59591 along with catalog IDs: 17180 and 19210, with no end time for continuous running."
    ): {
        "objective_uuid": "123e4567-e89b-12d3-a456-426614174000",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1000,
        "baseline_autonomy_rso": "17180,19210",
        "objective_end_time": None,
        "rso_id_list": ["11112", "99996", "59591"],
        "objective_name": "BaselineAutonomyObjective",
    },
}


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
    """Get fields from LLM response and ensure they're JSON-serializable."""
    displayable_fields = {}

    # Add response field first if it exists
    if hasattr(llm_response, "response"):
        displayable_fields["response"] = llm_response.response

    # Collect all other valid fields
    for field_name in dir(llm_response):
        if is_valid_field(field_name):
            value = getattr(llm_response, field_name)
            if not callable(value):  # Skip methods
                # Convert datetime objects to ISO format strings
                if isinstance(value, datetime):
                    displayable_fields[field_name] = value.isoformat()
                else:
                    displayable_fields[field_name] = value

    return displayable_fields


def get_current_iso_time() -> str:
    """Returns current time in ISO 8601 format with timezone (e.g., '2024-02-06T14:30:00+00:00')"""
    return datetime.now(pytz.UTC).isoformat()
