import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import requests

from app.interpretability_analysis import analyze_model
from app.utils import OBJECTIVE_TEST_CASES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_prompt(metadata):
    # Extract template from full_prompt in metadata
    template = metadata["full_prompt"].split("template='")[1].strip("'")

    # First unescape any \\n to \n
    template = template.replace("\\n", "\n")

    # Get the required values
    format_instructions = metadata["format_instructions"]
    query = metadata["input_text"]

    # Format the template with the values
    complete_prompt = template.format(
        format_instructions=format_instructions, query=query
    )

    return complete_prompt


def wait_for_server(url, timeout=120, interval=2):
    """Poll the server health endpoint until healthy or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.ok and response.json().get("status") == "healthy":
                print("Server is healthy.")
                return True
        except Exception:
            pass
        print("Waiting for server to be healthy...")
        time.sleep(interval)
    return False


def normalize_datetime_string(dt_str: str) -> str:
    """Extract datetime components from various string formats and return a normalized string."""
    # Patterns to match different datetime formats
    patterns = [
        # Matches strings like: datetime.datetime(2024, 8, 11, 19, 20, tzinfo=TzInfo(UTC))
        re.compile(
            r"(\d{4})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})[^\d]*(\d{1,2})"
        ),
        # Matches ISO strings like: 2024-08-11T19:20:00+00:00
        re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})"),
    ]

    for pattern in patterns:
        match = pattern.search(str(dt_str))
        if match:
            year, month, day, hour, minute = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}"
    return str(dt_str)


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
            # Convert expected datetime string to normalized format
            expected_norm = normalize_datetime_string(str(expected_value))
            # Convert predicted ISO format to normalized format
            current_norm = normalize_datetime_string(str(current_value))
            is_correct = expected_norm == current_norm

        # Handle numeric fields that could be int/float
        elif isinstance(expected_value, (int, float)):
            try:
                # Convert both to float for comparison
                expected_float = float(expected_value)
                current_float = float(current_value)
                # Compare with small tolerance for floating point
                is_correct = abs(expected_float - current_float) < 1e-10
            except (ValueError, TypeError):
                is_correct = False

        # Handle lists
        elif isinstance(expected_value, list):
            try:
                expected_sorted = sorted(str(x).strip() for x in expected_value)
                current_sorted = sorted(str(x).strip() for x in current_value)
                is_correct = expected_sorted == current_sorted
            except Exception:
                is_correct = False

        # Regular comparison
        else:
            is_correct = str(current_value).strip() == str(expected_value).strip()

        if is_correct:
            correct_fields += 1

        field_info["correct"] = is_correct
        field_details[field_name] = field_info

    accuracy = (correct_fields / total_fields) * 100 if total_fields > 0 else 0.0
    return accuracy, correct_fields, total_fields, field_details


def run_experiment(input_text, iterations, test_case=None):
    # Assume the FastAPI server is available at localhost:8888
    health_url = "http://localhost:8888/health"
    tracking_url = "http://localhost:8888/generate_experiment"

    if not wait_for_server(health_url):
        print("Server did not become healthy in time. Exiting.")
        sys.exit(1)

    # Define save_dir outside the loop so it's available later
    save_dir = "/app/plots"
    os.makedirs(save_dir, exist_ok=True)

    all_iters_data = {}

    # If test_case is provided, get the expected output
    expected_output = None
    test_prompt = None
    if test_case is not None:
        try:
            test_case = int(test_case)
            # Get the test case from OBJECTIVE_TEST_CASES
            test_cases = list(OBJECTIVE_TEST_CASES.items())
            if 1 <= test_case <= len(test_cases):
                test_prompt, expected_output = test_cases[test_case - 1]
                # Verify that INPUT_TEXT matches the test prompt
                if input_text.strip() != test_prompt.strip():
                    print(f"WARNING: INPUT_TEXT does not match test case {test_case}!")
                    print(f"Expected: {test_prompt}")
                    print(f"Received: {input_text}")
            else:
                print(
                    f"WARNING: Invalid TEST_CASE value: {test_case}. Must be between 1 and {len(test_cases)}."
                )
        except (ValueError, IndexError) as e:
            print(f"ERROR: Failed to process TEST_CASE: {e}")

    for iter_num in range(1, iterations + 1):
        print(f"--- Starting iteration {iter_num} ---")

        # Initialize default values to ensure we always have something to save
        workflow1_json = {"error": "API call not completed"}
        stats_1 = {"error": "Analysis not performed"}
        stats_2 = {"error": "Analysis not performed"}

        # 1. API Call with robust error handling
        try:
            payload = {"text": input_text}
            # Add timeout to prevent hanging
            resp = requests.post(tracking_url, json=payload, timeout=120)
            resp.raise_for_status()  # This will raise an exception for HTTP errors
            workflow1_json = resp.json()

            # Calculate accuracy metrics if expected_output is available
            if expected_output is not None and "llm_response" in workflow1_json:
                llm_response = workflow1_json["llm_response"]

                # Calculate objective type accuracy
                expected_obj_name = expected_output.get("objective_name")
                predicted_obj_name = llm_response.get("objective_name")
                objective_type_accuracy = (
                    1.0 if predicted_obj_name == expected_obj_name else 0.0
                )

                # Calculate field accuracy using the calculate_field_accuracy_custom function
                field_accuracy, correct_count, total_count, field_details = (
                    calculate_field_accuracy_custom(llm_response, expected_output)
                )

                # Add accuracy metrics to the workflow JSON
                workflow1_json["accuracy_metrics"] = {
                    "objective_type_accuracy": objective_type_accuracy,
                    "field_accuracy": field_accuracy,
                    "correct_fields": correct_count,
                    "total_fields": total_count,
                    "field_details": field_details,
                }

                print(
                    f"Iteration {iter_num} - Objective Type Accuracy: {objective_type_accuracy:.2%}"
                )
                print(
                    f"Iteration {iter_num} - Field Accuracy: {field_accuracy:.2f}% ({correct_count}/{total_count})"
                )

        except requests.exceptions.Timeout:
            print(f"Request timed out for iteration {iter_num}")
            workflow1_json = {"error": "Request timed out"}
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            workflow1_json = {"error": f"HTTP error: {e}"}
        except requests.exceptions.ConnectionError:
            print("Connection error - server may be down")
            workflow1_json = {"error": "Connection error - server may be down"}
        except json.JSONDecodeError:
            print("Invalid JSON in response")
            workflow1_json = {"error": "Invalid JSON in response"}
        except Exception as e:
            print(f"Error during API call: {e}")
            workflow1_json = {"error": f"Unexpected error: {str(e)}"}

        # 2. Extract prompts with safe navigation
        input_text_1 = None
        input_text_2 = None

        try:
            # Only attempt to extract prompts if we have valid data
            if "part_1" in workflow1_json and "part_2" in workflow1_json:
                part1 = workflow1_json.get("part_1", {})
                part2 = workflow1_json.get("part_2", {})

                # Log for debugging (useful if issues persist)
                logger.debug(f"Part 1 keys: {list(part1.keys())}")
                logger.debug(f"Part 2 keys: {list(part2.keys())}")

                metadata1 = part1.get("metadata", {})
                metadata2 = part2.get("metadata", {})

                # Only try to construct if all needed keys exist
                required_keys = ["full_prompt", "format_instructions", "input_text"]
                if all(key in metadata1 for key in required_keys):
                    input_text_1 = construct_prompt(metadata1)
                else:
                    logger.warning(
                        f"Missing required keys in metadata1: {metadata1.keys()}"
                    )

                if all(key in metadata2 for key in required_keys):
                    input_text_2 = construct_prompt(metadata2)
                else:
                    logger.warning(
                        f"Missing required keys in metadata2: {metadata2.keys()}"
                    )
        except Exception as e:
            print(f"Error extracting prompts: {e}")
            logger.exception("Prompt extraction failed")

        # 3. Run analysis for prompts that were successfully constructed
        try:
            if input_text_1:
                stats_1 = analyze_model(input_text_1)
            else:
                logger.warning(
                    "Skipping analysis for part_1: input_text_1 not available"
                )
        except Exception as e:
            print(f"Error during analysis for input_text_1: {e}")
            stats_1 = {"error": str(e)}

        try:
            if input_text_2:
                stats_2 = analyze_model(input_text_2)
            else:
                logger.warning(
                    "Skipping analysis for part_2: input_text_2 not available"
                )
        except Exception as e:
            print(f"Error during analysis for input_text_2: {e}")
            stats_2 = {"error": str(e)}

        # 4. Always create and save iteration data
        iter_data = {
            "workflow1": workflow1_json,
            "workflow2": {"part_1": stats_1, "part_2": stats_2},
            "iteration_number": iter_num,
            "timestamp": datetime.now().isoformat(),
            "input_text": input_text,
            "status": "error" if "error" in workflow1_json else "success",
        }

        try:
            iter_filename = f"iter_{iter_num}.json"
            with open(os.path.join(save_dir, iter_filename), "w") as f:
                json.dump(iter_data, f, indent=2)
            print(f"Saved iteration {iter_num} data to {iter_filename}")
        except Exception as save_err:
            print(f"Error saving iteration data: {save_err}")
            logger.exception(f"Failed to save iteration {iter_num} data")

        # Always add to master data set
        all_iters_data[f"iter_{iter_num}"] = iter_data

        # Brief pause between iterations to reduce load on server
        if iter_num < iterations:
            time.sleep(1)

    # 5. Save final consolidated results
    try:
        current_time = datetime.now()
        timestamp = current_time.strftime("%m%d%Y_%H%M")
        final_filename = f"complete_experiment_{timestamp}.json"
        with open(os.path.join(save_dir, final_filename), "w") as f:
            json.dump(all_iters_data, f, indent=2)
        print(
            f"SUCCESS: All {iterations} iterations completed. Saved JSON to {final_filename}"
        )
    except Exception as final_err:
        print(f"ERROR: Failed to save final data: {final_err}")
        logger.exception("Failed to save final experiment data")

        # Try emergency backup
        try:
            backup_path = os.path.join(save_dir, "backup_experiment.json")
            with open(backup_path, "w") as f:
                json.dump(all_iters_data, f, indent=2)
            print(f"Emergency backup saved as {backup_path}")
        except Exception as backup_err:
            print(f"CRITICAL ERROR: Could not save results anywhere! {backup_err}")
            logger.critical("Failed to save backup experiment data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run merged experiment workflow.")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be sent to the /generate_full_objective_tracking endpoint.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of experiment iterations to run.",
    )
    parser.add_argument(
        "--test_case",
        type=int,
        default=None,
        help="Test case number to use for accuracy evaluation.",
    )
    args = parser.parse_args()

    # Get TEST_CASE from environment variable if not provided as argument
    test_case = args.test_case
    if test_case is None and "TEST_CASE" in os.environ:
        try:
            test_case = int(os.environ["TEST_CASE"])
        except ValueError:
            print(
                f"WARNING: Invalid TEST_CASE environment variable: {os.environ['TEST_CASE']}"
            )

    run_experiment(args.input_text, args.iterations, test_case)
