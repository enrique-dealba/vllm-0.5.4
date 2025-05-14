import asyncio
import datetime
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Benchmark settings
N_ITERATIONS = 1  # Number of times to run each test case
RATE_LIMIT_DELAY = 2.0  # Seconds to wait between requests
MAX_CONCURRENT_REQUESTS = 1  # Maximum number of concurrent requests. Note: Must be 1

# ------------------------------------------------------------------------------
# Test cases: Each key is a prompt; each value is the expected output dictionary.
# (The expected output is taken from your examples.)
# ------------------------------------------------------------------------------
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
    # (
    #     "Create a GeodssRevisitObjective for targets 12345,67890 using sensors RME15,LMNT17. Set U//FOUO marking, REAL mode, "
    #     "priority 10, RATE_TRACK_SIDEREAL tracking. Start at 2024-08-11 19:20:00+00:00. Set readout_rate 1 (2MHz), gain_setting 0 (High Gain), "
    #     "soi_filter 1 (1% Light), auto_track_type 1 (Automatic), camera_mode 0 (Normal), array_kind 0 (Main), binning_mode 1 (HW Binning), "
    #     "scan_mode 1 (Single Frame)."
    # ): {
    #     "acquisition_type": 0,
    #     "array_kind": 0,
    #     "auto_track_roi_position": 0,
    #     "auto_track_type": 1,
    #     "binning_mode": 1,
    #     "camera_mode": 0,
    #     "classification_marking": "U//FOUO",
    #     "collect_request_type": "RATE_TRACK_SIDEREAL",
    #     "command": 0,
    #     "data_mode": "REAL",
    #     "frame_type": "LIGHT",
    #     "gain_setting": 0,
    #     "ignore_other_objective_intent_submissions": False,
    #     "integration_time": None,
    #     "intent_end_time": None,
    #     "intent_start_time": None,
    #     "num_observations": 1,
    #     "num_skip_frames": 0,
    #     "number_of_frames": None,
    #     "objective_end_time": None,
    #     "objective_name": "GeodssRevisitObjective",
    #     "objective_start_time": "datetime.datetime(2024, 8, 11, 19, 20, tzinfo=TzInfo(UTC))",
    #     "objective_uuid": None,
    #     "observation_interval": 0,
    #     "optimal_frames_per_hour": 400,
    #     "overscan": 0,
    #     "patience_minutes": 30,
    #     "priority": 10,
    #     "rate_track_verify": 0,
    #     "readout_rate_setting": 1,
    #     "revisits_per_hour": None,
    #     "scan_mode": 1,
    #     "sensor_name_list": ["RME15", "LMNT17"],
    #     "soi_filter_position": 1,
    #     "target_id_list": ["12345", "67890"],
    #     "visibility_check": False,
    # },
    # # Example 4: PeriodicRevisitObjective
    # (
    #     "Create a PeriodicRevisitObjective for targets 12225,68887 using sensors RME05,LMNT06. "
    #     "Set S marking, TEST mode, priority 2, patience minutes 30, ignore other objective submissions false. "
    #     "Start objective at 2024-06-21 19:20:00+00:00. Set optimal frames per hour 400, number of frames 5, integration time 2 seconds."
    # ): {
    #     "classification_marking": "S",
    #     "target_id_list": ["12225", "68887"],
    #     "sensor_name_list": ["RME05", "LMNT06"],
    #     "data_mode": "TEST",
    #     "collect_request_type": "RATE_TRACK_SIDEREAL",
    #     "frame_type": "LIGHT",
    #     "visibility_check": False,
    #     "patience_minutes": 30,
    #     "revisits_per_hour": None,
    #     "number_of_frames": 5,
    #     "integration_time": 2,
    #     "binning": None,
    #     "objective_start_time": "datetime.datetime(2024, 6, 21, 19, 20, tzinfo=TzInfo(UTC))",
    #     "objective_end_time": None,
    #     "priority": 2,
    #     "ignore_other_objective_intent_submissions": False,
    #     "optimal_frames_per_hour": 400,
    #     "objective_uuid": None,
    #     "intent_start_time": None,
    #     "intent_end_time": None,
    #     "objective_name": "PeriodicRevisitObjective",
    # },
    # # Example 5: UctObservationObjective
    # (
    #     "Create a UctObservationObjective for UCT RSOs 12345,67890 using sensors RME18, LMNT19. "
    #     "Set U//FOUO marking, REAL mode, GEO regime, priority 10, 6 revisits per hour. "
    #     "Start at 2024-09-12 19:20:00+00:00. Enable sorting by brightest UCT, set end time offset to 60 minutes, visibility check true. "
    #     "Set number of frames to 5 and integration time 2 seconds."
    # ): {
    #     "classification_marking": "U//FOUO",
    #     "uct_rso_id_list": ["12345", "67890"],
    #     "sensor_name_list": ["RME18", "LMNT19"],
    #     "data_mode": "REAL",
    #     "collect_request_type": "RATE_TRACK_SIDEREAL",
    #     "frame_type": "LIGHT",
    #     "orbital_regime": "GEO",
    #     "visibility_check": True,
    #     "patience_minutes": 30,
    #     "revisits_per_hour": 6.0,
    #     "number_of_frames": 5,
    #     "integration_time": 2,
    #     "binning": None,
    #     "end_time_offset_minutes": 60,
    #     "objective_uuid": None,
    #     "objective_start_time": "datetime.datetime(2024, 9, 12, 19, 20, tzinfo=TzInfo(UTC))",
    #     "objective_end_time": None,
    #     "priority": 10,
    #     "sort_by_brightest_uct": True,
    #     "objective_name": "UctObservationObjective",
    # },
    # # Example 6: SingleIntentObjective
    # (
    #     "Create a SingleIntentObjective with target ID 11223, RSO ID 66778, using sensors RME22,LMNT24. "
    #     "Set U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, priority 10. "
    #     "Start objective at 2024-10-01 11:20:00+00:00. Set number of frames to 5, integration time 2 seconds, binning 2."
    # ): {
    #     "classification_marking": "U",
    #     "target_id": "11223",
    #     "rso_id": "66778",
    #     "sensor_name_list": ["RME22", "LMNT24"],
    #     "data_mode": "REAL",
    #     "collect_request_type": "RATE_TRACK_SIDEREAL",
    #     "frame_type": "LIGHT",
    #     "number_of_frames": 5,
    #     "integration_time": 2,
    #     "priority": 10,
    #     "binning": 2,
    #     "intent_start_time": None,
    #     "intent_end_time": None,
    #     "objective_uuid": None,
    #     "objective_start_time": "datetime.datetime(2024, 10, 1, 11, 20, tzinfo=TzInfo(UTC))",
    #     "objective_end_time": None,
    #     "objective_name": "SingleIntentObjective",
    # },
    # # Example 7: DataEnrichmentObjective
    # (
    #     "Create a DataEnrichmentObjective for targets 55441, 99886, 50051 using sensors RME31,LMNT34. "
    #     "Set U//FOUO marking, REAL mode, RATE_TRACK tracking, and set max RSO to observe as 8, 10 revisits per hour. "
    #     "Start at 2025-01-21 08:00:00+00:00. Set visibility check true."
    # ): {
    #     "classification_marking": "U//FOUO",
    #     "data_mode": "REAL",
    #     "objective_uuid": None,
    #     "target_id_list": ["55441", "99886", "50051"],
    #     "sensor_name_list": ["RME31", "LMNT34"],
    #     "collect_request_type": "RATE_TRACK",  # overridden to RATE_TRACK
    #     "frame_type": "LIGHT",
    #     "binning": None,
    #     "max_rso_to_observe": 8,
    #     "revisits_per_hour": 10,
    #     "objective_start_time": "datetime.datetime(2025, 1, 21, 8, 0, tzinfo=TzInfo(UTC))",
    #     "objective_end_time": None,
    #     "priority": 20,
    #     "intent_start_time": None,
    #     "intent_end_time": None,
    #     "visibility_check": True,
    #     "objective_name": "DataEnrichmentObjective",
    # },
    # # Example 8: SensorCheckoutObjective
    # (
    #     "Create a SensorCheckoutObjective with classification_marking='U' and sensor_name='RME01'. "
    #     "Set data_mode='REAL', orbital_regime='GEO', collect_request_type='RATE_TRACK_SIDEREAL', priority=10, "
    #     "revisits_per_hour=1.0, objective_start_time='2025-02-01 19:20:00+00:00', visibility_check=true, "
    #     "patience_minutes=30, number_of_frames=5, integration_time=2."
    # ): {
    #     "classification_marking": "U",
    #     "sensor_name": "RME01",
    #     "orbital_regime": "GEO",
    #     "data_mode": "REAL",
    #     "collect_request_type": "RATE_TRACK_SIDEREAL",
    #     "frame_type": "LIGHT",
    #     "visibility_check": True,
    #     "patience_minutes": 30,
    #     "revisits_per_hour": 1.0,
    #     "number_of_frames": 5,
    #     "integration_time": 2,
    #     "binning": None,
    #     "objective_uuid": None,
    #     "objective_start_time": "datetime.datetime(2025, 2, 1, 19, 20, tzinfo=TzInfo(UTC))",
    #     "objective_end_time": None,
    #     "intent_start_time": None,
    #     "intent_end_time": None,
    #     "priority": 10,
    #     "objective_name": "SensorCheckoutObjective",
    # },
    # # Example 9: BaselineAutonomyObjective
    # (
    #     "Create a BaselineAutonomyObjective with UUID '123e4567-e89b-12d3-a456-426614174000'. "
    #     "Use U markings, REAL mode, LIGHT frame type, priority 1000. "
    #     "Use following RSO ids: 11112, 99996, and 59591 along with catalog IDs: 17180 and 19210, with no end time for continuous running."
    # ): {
    #     "objective_uuid": "123e4567-e89b-12d3-a456-426614174000",
    #     "classification_marking": "U",
    #     "data_mode": "REAL",
    #     "frame_type": "LIGHT",
    #     "priority": 1000,
    #     "baseline_autonomy_rso": "17180,19210",
    #     "objective_end_time": None,
    #     "rso_id_list": ["11112", "99996", "59591"],
    #     "objective_name": "BaselineAutonomyObjective",
    # },
}


# Precompile regex patterns
_PATTERNS = [
    # Matches strings like: datetime.datetime(2024, 8, 11, 19, 20, tzinfo=TzInfo(UTC))
    re.compile(r"(\d{4})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})"),
    # Matches ISO strings like: 2024-08-11T19:20:00+00:00
    re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})"),
]


def normalize_datetime_string(dt_str: str) -> str:
    """Extract datetime components from various string formats and return a normalized string in
    the format "YYYY-MM-DD HH:MM".
    """
    for pattern in _PATTERNS:
        match = pattern.search(dt_str)
        if match:
            year, month, day, hour, minute = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}"
    return dt_str


def calculate_field_accuracy_custom(
    predicted: dict, expected: dict
) -> Tuple[float, int, int, Dict[str, Dict[str, Any]]]:
    """Compare fields and return detailed field comparison info."""
    correct_fields = 0
    total_fields = len(expected)
    field_details = {}

    for field_name, expected_value in expected.items():
        field_info = {
            "expected": expected_value,
            "predicted": predicted.get(field_name, "MISSING"),
            "correct": False,
        }

        if field_name not in predicted:
            field_details[field_name] = field_info
            continue

        current_value = predicted[field_name]
        field_info["predicted"] = current_value

        # Handle datetime fields
        if field_name in [
            "objective_start_time",
            "objective_end_time",
            "search_start_time",
        ]:
            expected_norm = normalize_datetime_string(str(expected_value))
            current_norm = normalize_datetime_string(str(current_value))
            is_correct = expected_norm == current_norm

        elif isinstance(expected_value, (int, float)):
            try:
                expected_float = float(expected_value)
                current_float = float(current_value)
                is_correct = abs(expected_float - current_float) < 1e-10
            except Exception:
                is_correct = False

        elif isinstance(expected_value, list):
            try:
                expected_sorted = sorted(str(x).strip() for x in expected_value)
                current_sorted = sorted(str(x).strip() for x in current_value)
                is_correct = expected_sorted == current_sorted
            except Exception:
                is_correct = False

        else:
            is_correct = str(current_value).strip() == str(expected_value).strip()

        if is_correct:
            correct_fields += 1

        field_info["correct"] = is_correct
        field_details[field_name] = field_info

    accuracy = (correct_fields / total_fields) * 100 if total_fields > 0 else 0.0
    return accuracy, correct_fields, total_fields, field_details


def calculate_slot_metrics(predicted: dict, expected: dict) -> Dict[str, float]:
    """Calculate slot-level precision, recall, and F1 for field presence."""
    expected_fields = set(expected.keys())
    predicted_fields = set(predicted.keys())
    tp = len(expected_fields & predicted_fields)
    fp = len(predicted_fields - expected_fields)
    fn = len(expected_fields - predicted_fields)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class ObjectiveBenchmark:
    def __init__(self, api_url: str = "http://localhost:8888"):
        self.api_url = api_url
        self.model_name = "None"
        self.results = []
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def test_single_objective(self, query: str, expected: dict) -> Dict:
        async with self.semaphore:
            try:
                async with httpx.AsyncClient() as client:
                    start_time = time.time()
                    response = await client.post(
                        f"{self.api_url}/generate_full_objective",
                        json={"text": query},
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    execution_time = time.time() - start_time

                    # Metric 1: Exact Match
                    exact_match = json.dumps(result, sort_keys=True) == json.dumps(
                        expected, sort_keys=True
                    )

                    # Field-level detailed metrics
                    field_accuracy, correct_count, total_count, field_details = (
                        calculate_field_accuracy_custom(result, expected)
                    )
                    # Slot-level presence metrics
                    slot_metrics = calculate_slot_metrics(result, expected)

                    predicted_obj_name = result.get("objective_name")
                    expected_obj_name = expected.get("objective_name")
                    objective_name_correct = predicted_obj_name == expected_obj_name

                    await asyncio.sleep(RATE_LIMIT_DELAY)

                    return {
                        "query": query,
                        "exact_match": exact_match,
                        "slot_metrics": slot_metrics,
                        "expected_objective_name": expected_obj_name,
                        "predicted_objective_name": predicted_obj_name,
                        "objective_name_correct": objective_name_correct,
                        "field_accuracy": field_accuracy,
                        "correct_field_count": correct_count,
                        "total_field_count": total_count,
                        "execution_time": execution_time,
                        "field_details": field_details,
                        "predicted": result,
                    }
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                await asyncio.sleep(RATE_LIMIT_DELAY)
                return {
                    "query": query,
                    "exact_match": False,
                    "slot_metrics": {
                        "tp": 0,
                        "fp": 0,
                        "fn": len(expected),
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                    },
                    "expected_objective_name": expected.get("objective_name"),
                    "predicted_objective_name": "ERROR",
                    "objective_name_correct": False,
                    "field_accuracy": 0.0,
                    "correct_field_count": 0,
                    "total_field_count": len(expected),
                    "execution_time": 0,
                    "field_details": {},
                    "predicted": {},
                }

    async def run_benchmark(self):
        """Run the benchmark on all test cases over multiple iterations."""
        all_results = []
        for iteration in range(N_ITERATIONS):
            logger.info(f"Starting iteration {iteration + 1}/{N_ITERATIONS}")
            tasks = [
                self.test_single_objective(query, expected)
                for query, expected in OBJECTIVE_TEST_CASES.items()
            ]
            iteration_results = await asyncio.gather(*tasks)
            all_results.extend(iteration_results)
            if iteration < N_ITERATIONS - 1:
                await asyncio.sleep(RATE_LIMIT_DELAY * 2)
        self.results = all_results
        return self.results

    def print_results_analysis(self):
        """Print comprehensive analysis of benchmark results with cumulative field accuracy."""
        df = pd.DataFrame(self.results)

        # Calculate overall metrics
        total_tests = len(df)
        predicted_names = df["predicted_objective_name"].fillna("NONE")
        expected_names = df["expected_objective_name"].fillna("NONE")
        accuracy = accuracy_score(expected_names, predicted_names)
        obj_accuracy = accuracy
        avg_execution_time = df["execution_time"].mean()

        # New top-level metrics
        exact_match_rate = df["exact_match"].mean()
        # slot_metrics is a column of dicts; turn it into a DataFrame
        slot_df = pd.DataFrame(df["slot_metrics"].tolist())
        avg_precision = slot_df["precision"].mean()
        avg_recall = slot_df["recall"].mean()
        avg_f1 = slot_df["f1"].mean()

        print("\n=== Benchmark Summary ===")
        print(f"Exact Match Rate:       {exact_match_rate:.2%}")
        print(
            "Slot-Level (presence) — "
            f"P: {avg_precision:.2%}, "
            f"R: {avg_recall:.2%}, "
            f"F1: {avg_f1:.2%}"
        )
        print(f"Total Test Cases: {len(OBJECTIVE_TEST_CASES)}")
        print(f"Iterations per Test Case: {N_ITERATIONS}")
        print(f"Total Tests Run: {total_tests}")
        print(f"Overall Objective Accuracy: {obj_accuracy:.2%}")
        print(f"Average Execution Time: {avg_execution_time:.3f}s")

        # Track field-specific accuracy across all tests
        field_accuracy_counts = {}  # {field_name: (correct_count, total_count)}

        # Group results and print per-objective stats
        grouped_results = df.groupby("expected_objective_name").agg(
            {
                "predicted_objective_name": "first",
                "objective_name_correct": "all",
                "correct_field_count": "sum",
                "total_field_count": "sum",
                "execution_time": "mean",
                "field_details": list,
            }
        )

        print("\n=== Detailed Results per Objective Type ===")
        for obj_name, row in grouped_results.iterrows():
            print("-" * 80)
            print(f"Expected Objective Name: {obj_name}")
            print(
                f"Predicted Objective Name: {row['predicted_objective_name']} "
                f"({'Correct' if row['objective_name_correct'] else 'Incorrect'})"
            )

            total_correct = row["correct_field_count"]
            total_fields = row["total_field_count"]
            cumulative_accuracy = (
                (total_correct / total_fields * 100) if total_fields > 0 else 0
            )
            sub = df[df.expected_objective_name == obj_name]
            slot_sub = pd.DataFrame(sub["slot_metrics"].tolist())
            print(
                "  Slot-P/R/F1:",
                f"{slot_sub.precision.mean():.2%}/"
                f"{slot_sub.recall.mean():.2%}/"
                f"{slot_sub.f1.mean():.2%}",
            )

            print(
                f"Field Accuracy: {cumulative_accuracy:.2f}% "
                f"({total_correct}/{total_fields})"
            )

            # Only show incorrect fields
            print("\nIncorrect Fields:")
            for iteration_details in row["field_details"]:
                for field_name, details in sorted(iteration_details.items()):
                    # Update field accuracy tracking
                    if field_name not in field_accuracy_counts:
                        field_accuracy_counts[field_name] = [0, 0]
                    field_accuracy_counts[field_name][1] += 1  # increment total
                    if details["correct"]:
                        field_accuracy_counts[field_name][0] += 1  # increment correct

                    # Show only incorrect fields
                    if not details["correct"]:
                        print(f"\n{field_name}:")
                        print(f"  Expected: {details['expected']}")
                        print(f"  Predicted: {details['predicted']}")

            print(f"\nExecution Time: {row['execution_time']:.3f}s")

        # Print overall field-specific accuracy for non-perfect fields
        print("\n=== Field-Specific Accuracy ===")
        total_correct_all_fields = 0
        total_fields_all = 0

        for field_name, (correct, total) in sorted(field_accuracy_counts.items()):
            accuracy = (correct / total) * 100
            total_correct_all_fields += correct
            total_fields_all += total
            if accuracy < 100:  # Only show non-perfect fields
                print(f"{field_name}: {accuracy:.2f}%")

        # Add overall field accuracy across all types
        overall_field_accuracy = (
            (total_correct_all_fields / total_fields_all * 100)
            if total_fields_all > 0
            else 0
        )
        print("\n=== Overall Objective Accuracy ===")
        print(f"Accuracy: {obj_accuracy:.2%}")
        print("\n=== Overall Field Accuracy ===")
        print(
            f"Total Accuracy Across All Fields: {overall_field_accuracy:.2f}% ({total_correct_all_fields}/{total_fields_all})"
        )

    async def generate_results_report(self) -> dict:
        """Build a comprehensive JSON report of the last benchmark run."""
        # 1) Pull application settings
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.api_url}/settings")
            resp.raise_for_status()
            settings = resp.json()
            self.model_name = settings["LLM_MODEL_NAME"]

        # 2) Timestamp
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d%Y%H%M")

        # 3) Metrics dataframe
        df = pd.DataFrame(self.results)

        # Overall objective accuracy
        expected = df["expected_objective_name"].fillna("NONE")
        predicted = df["predicted_objective_name"].fillna("NONE")
        objective_accuracy = accuracy_score(expected, predicted)

        # Top-level metrics
        total_tests = len(df)
        exact_match_rate = df["exact_match"].mean()
        schema_valid_rate = df["schema_valid"].mean()
        avg_exec_time = df["execution_time"].mean()

        # Slot‐level (presence) metrics
        slot_df = pd.DataFrame(df["slot_metrics"].tolist())
        avg_precision = slot_df["precision"].mean()
        avg_recall = slot_df["recall"].mean()
        avg_f1 = slot_df["f1"].mean()

        # Field‐specific accuracy counts
        field_counts = {}
        for r in self.results:
            for field, info in r["field_details"].items():
                correct, total = field_counts.get(field, [0, 0])
                total += 1
                correct += int(info["correct"])
                field_counts[field] = [correct, total]

        field_specific = {
            field: correct / total for field, (correct, total) in field_counts.items()
        }
        overall_field_accuracy = sum(c for c, _ in field_counts.values()) / sum(
            t for _, t in field_counts.values()
        )

        # Detailed per‐objective breakdown
        grouped = df.groupby("expected_objective_name")
        details = {}
        for obj_name, subdf in grouped:
            # name‐level stats
            pred_name = subdf["predicted_objective_name"].iat[0]
            name_correct = bool(subdf["objective_name_correct"].all())

            # slot metrics for this objective
            slot_sub = pd.DataFrame(subdf["slot_metrics"].tolist())
            slot_stats = {
                "precision": slot_sub["precision"].mean(),
                "recall": slot_sub["recall"].mean(),
                "f1": slot_sub["f1"].mean(),
            }

            # cumulative field accuracy
            total_correct = subdf["correct_field_count"].sum()
            total_fields = subdf["total_field_count"].sum()
            field_acc = total_correct / total_fields if total_fields else None

            # error catalog
            errors = []
            for run_details in subdf["field_details"]:
                for field, info in run_details.items():
                    if not info["correct"]:
                        errors.append(
                            {
                                "field": field,
                                "expected": info["expected"],
                                "predicted": info["predicted"],
                            }
                        )

            details[obj_name] = {
                "predicted_name": pred_name,
                "name_correct": name_correct,
                "slot_metrics": slot_stats,
                "field_accuracy": field_acc,
                "errors": errors,
                "average_latency_s": subdf["execution_time"].mean(),
            }

        report = {
            "generated_at": now,
            "settings": settings,
            "benchmark_configuration": {
                "test_cases": len(OBJECTIVE_TEST_CASES),
                "iterations": N_ITERATIONS,
                "total_requests": total_tests,
            },
            "summary_metrics": {
                "objective_accuracy": objective_accuracy,
                "exact_match_rate": exact_match_rate,
                "schema_valid_rate": schema_valid_rate,
                "avg_latency_s": avg_exec_time,
                "slot_presence": {
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1": avg_f1,
                },
            },
            "field_specific_accuracy": field_specific,
            "overall_field_accuracy": overall_field_accuracy,
            "per_objective_details": details,
        }

        return report


async def main():
    start_time = time.time()
    api_url = "http://localhost:8888"

    # Health check with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                health_response = await client.get(f"{api_url}/health")
                if health_response.status_code == 200:
                    logger.info("API health check passed.")
                    break
                else:
                    logger.warning(
                        f"API not healthy (attempt {attempt + 1}/{max_retries})"
                    )
            await asyncio.sleep(5)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Could not connect to API after {max_retries} attempts: {e}"
                )
                return
            await asyncio.sleep(5)

    # Run the full objective benchmark
    benchmark = ObjectiveBenchmark(api_url)
    await benchmark.run_benchmark()

    # Generate the JSON report
    report = await benchmark.generate_results_report()

    # Determine model name and timestamp for filename
    model_name = benchmark.model_name or report.get("settings", {}).get(
        "LLM_MODEL_NAME", "model"
    )
    # sanitize e.g. replace slashes or at-signs
    model_name = model_name.replace("/", "_").replace("@", "_")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d%Y%H%M")
    filename = f"{model_name}_{timestamp}.json"

    # Ensure output directory exists
    out_dir = Path("../May2025_JSONs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write report
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {out_path.resolve()}")

    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"Total Benchmark Time: {minutes}mins {seconds}secs")


if __name__ == "__main__":
    asyncio.run(main())
