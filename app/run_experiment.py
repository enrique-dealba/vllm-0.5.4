import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

from app.interpretability_analysis import analyze_model


def construct_prompt(metadata):
    try:
        template = metadata["full_prompt"].split("template='")[1].strip("'")
    except IndexError:
        template = metadata["full_prompt"]
    template = template.replace("\\n", "\n")
    format_instructions = metadata["format_instructions"]
    query = metadata["input_text"]
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


def run_experiment(input_text, iterations):
    # Assume the FastAPI server is available at localhost:8888
    health_url = "http://localhost:8888/health"
    tracking_url = "http://localhost:8888/generate_full_objective_tracking"

    if not wait_for_server(health_url):
        print("Server did not become healthy in time. Exiting.")
        sys.exit(1)

    all_iters_data = {}

    for iter_num in range(1, iterations + 1):
        print(f"--- Starting iteration {iter_num} ---")
        # 1. Call the tracking endpoint with the input_text
        payload = {"text": input_text}
        try:
            resp = requests.post(tracking_url, json=payload)
            resp.raise_for_status()
            workflow1_json = (
                resp.json()
            )  # Expected to contain keys "part_1" and "part_2"
        except Exception as e:
            print(f"Error during API call: {e}")
            continue

        # 2. Extract prompts from the workflow1 JSON using construct_prompt
        try:
            metadata1 = workflow1_json.get("part_1", {}).get("metadata", {})
            metadata2 = workflow1_json.get("part_2", {}).get("metadata", {})
            input_text_1 = construct_prompt(metadata1)
            input_text_2 = construct_prompt(metadata2)
        except Exception as e:
            print(f"Error constructing prompts: {e}")
            continue

        # 3. Run analysis (workflow2) for both prompts.
        try:
            stats_1 = analyze_model(input_text_1)
        except Exception as e:
            print(f"Error during analysis for input_text_1: {e}")
            stats_1 = {"error": str(e)}

        try:
            stats_2 = analyze_model(input_text_2)
        except Exception as e:
            print(f"Error during analysis for input_text_2: {e}")
            stats_2 = {"error": str(e)}

        # 4. Combine both workflow JSONs into one for this iteration.
        iter_data = {
            "workflow1": workflow1_json,
            "workflow2": {"part_1": stats_1, "part_2": stats_2},
        }
        # Save individual iteration file
        iter_filename = f"iter_{iter_num}.json"
        save_dir = "/app/plots"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, iter_filename), "w") as f:
            json.dump(iter_data, f, indent=2)
        print(f"Saved iteration {iter_num} data to {iter_filename}")
        all_iters_data[f"iter_{iter_num}"] = iter_data

    # 5. Consolidate all iterations into one final JSON.
    current_time = datetime.now()
    timestamp = current_time.strftime("%m%d%Y_%H%M")
    final_filename = f"complete_experiment_{timestamp}.json"
    with open(os.path.join(save_dir, final_filename), "w") as f:
        json.dump(all_iters_data, f, indent=2)
    print(
        f"SUCCESS: All {iterations} iterations completed. Saved JSON to {final_filename}"
    )


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
    args = parser.parse_args()
    run_experiment(args.input_text, args.iterations)
