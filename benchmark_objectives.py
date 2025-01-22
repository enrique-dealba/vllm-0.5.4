import asyncio
import json
import logging
from typing import Dict

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tests.objective_test_cases import OBJECTIVE_TEST_CASES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectiveBenchmark:
    def __init__(self, api_url: str = "http://0.0.0.0:8888"):
        self.api_url = api_url
        self.results = []

    async def test_single_objective(self, query: str, expected: str) -> Dict:
        """Test a single objective query against the API."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_url}/generate_objective",
                    json={"text": query},
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                # Extract objective_name from response
                predicted = result.get("objective_name", "")
                execution_time = result.get("execution_time_seconds", 0)

                return {
                    "query": query,
                    "expected": expected,
                    "predicted": predicted,
                    "execution_time": execution_time,
                    "correct": expected == predicted,
                }
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                return {
                    "query": query,
                    "expected": expected,
                    "predicted": "ERROR",
                    "execution_time": 0,
                    "correct": False,
                }

    async def run_benchmark(self):
        """Run benchmark on all test cases."""
        tasks = []
        for query, expected in OBJECTIVE_TEST_CASES.items():
            tasks.append(self.test_single_objective(query, expected))

        self.results = await asyncio.gather(*tasks)
        return self.results

    def analyze_results(self):
        """Analyze benchmark results and generate metrics."""
        df = pd.DataFrame(self.results)

        total_tests = len(df)
        successful_tests = len(df[df["correct"]])
        accuracy = accuracy_score(df["expected"], df["predicted"])
        avg_execution_time = df["execution_time"].mean()

        report = classification_report(
            df["expected"], df["predicted"], output_dict=True
        )

        labels = list(set(OBJECTIVE_TEST_CASES.values()))
        cm = confusion_matrix(df["expected"], df["predicted"], labels=labels)

        confusion_matrix_data = {"matrix": cm.tolist(), "labels": labels}

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "accuracy": accuracy,
            "avg_execution_time": avg_execution_time,
            "detailed_report": report,
            "confusion_matrix": confusion_matrix_data,
            "results_df": df,
        }


def save_results(analysis_results: Dict, output_file: str = "benchmark_results.json"):
    """Save benchmark results to file."""
    analysis_results["results_df"] = analysis_results["results_df"].to_dict("records")

    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)


def print_ascii_confusion_matrix(cm_data: Dict):
    """Print a text-based visualization of the confusion matrix."""
    matrix = cm_data["matrix"]
    labels = cm_data["labels"]

    max_label_width = max(len(label) for label in labels)
    max_num_width = max(len(str(num)) for row in matrix for num in row)

    print("\nConfusion Matrix:")
    print(" " * (max_label_width + 2) + "Predicted")
    print(" " * (max_label_width + 2) + "─" * (len(labels) * (max_num_width + 1)))

    print(" " * (max_label_width + 2), end="")
    for label in labels:
        print(f"{label[:max_num_width]:{max_num_width}}", end=" ")
    print(
        "\n" + " " * (max_label_width + 2) + "─" * (len(labels) * (max_num_width + 1))
    )

    for i, label in enumerate(labels):
        print(f"{label:{max_label_width}} │", end=" ")
        for j in range(len(labels)):
            print(f"{matrix[i][j]:{max_num_width}}", end=" ")
        print()


async def main():
    # API health check
    async with httpx.AsyncClient() as client:
        try:
            health_response = await client.get("http://localhost:8888/health")
            if health_response.status_code != 200:
                logger.error("API is not healthy. Exiting.")
                return
        except Exception as e:
            logger.error(f"Could not connect to API: {e}")
            return

    # Run benchmark
    benchmark = ObjectiveBenchmark()
    await benchmark.run_benchmark()

    analysis = benchmark.analyze_results()
    save_results(analysis)

    print("\n=== Benchmark Results ===")
    print(f"Total Tests: {analysis['total_tests']}")
    print(f"Successful Tests: {analysis['successful_tests']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print(f"Average Execution Time: {analysis['avg_execution_time']:.3f}s")
    print("\nDetailed Classification Report:")
    print(pd.DataFrame(analysis["detailed_report"]).transpose())

    print_ascii_confusion_matrix(analysis["confusion_matrix"])


if __name__ == "__main__":
    asyncio.run(main())
