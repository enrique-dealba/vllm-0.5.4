import asyncio
import logging
from typing import Dict

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_ITERATIONS = 2  # Num of times to run each test case
RATE_LIMIT_DELAY = 2.0  # Seconds between requests
MAX_CONCURRENT_REQUESTS = 3  # Maximum number of concurrent requests

OBJECTIVE_TEST_CASES = {
    # PeriodicRevisitObjective
    "Track object 44248 with sensors RME01 and LMNT45, revisiting twice per hour for the next 36 hours using TEST mode, 'S' markings, and set priority to 2. Begin at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. Use RATE_TRACK_SIDEREAL as the collect request type and operate in LEO orbital regime, and set number of frames to 5 and integration time to 2 seconds.": "PeriodicRevisitObjective",
    "Track celestial object 21212 with sensors RME33, ABQ42, using REAL mode, revisiting four times per hour, starting execution for a 48-hour plan, marked as 'U//FOUO', with priority set to 1. Begins on 2024-05-21 19:20:00.150000+00:00 and finishes by 2024-05-21 22:30:00.250000+00:00. Employ RATE_TRACK for tracking type and GSO for orbital regime, and set number of frames to 10 and integration time to 1 second.": "PeriodicRevisitObjective",
    "Track RSO object 43567 using sensors ABQ42, UKR88 in TEST mode, schedule four revisits per hour, initiate with a 36-hour plan, marked as 'C', with priority set to 3. Start at 2024-05-21 19:20:00.150000+00:00, ending at 2024-05-21 22:30:00.250000+00:00. Use SIDEREAL tracking and MEO orbital regime, and set number of frames to 12 and integration time to 4 seconds.": "PeriodicRevisitObjective",
    "Monitor celestial object 20394 with sensors UKR88, RME02 in REAL mode, configure five revisits per hour, begin with a 42-hour schedule, classified as 'U//FOUO', priority level 3. Starts at 2024-05-21 19:20:00.150000+00:00 and concludes at 2024-05-21 22:30:00.250000+00:00. Apply RATE_TRACK_SIDEREAL and operate in HEO orbital regime, and set number of frames to 3 and integration time to 10 seconds.": "PeriodicRevisitObjective",
    "Observe object 31705 with sensors LMNT33, RME99 in REAL mode, perform three revisits per hour, initiate with a 24-hour strategy, marked as 'S', with priority level 1. Begins on 2024-05-21 19:20:00.150000+00:00 and ends by 2024-05-21 22:30:00.250000+00:00. Set collect request type to RATE_TRACK and orbital regime to GSO, and set number of frames to 16 and integration time to 20 seconds.": "PeriodicRevisitObjective",
    "Follow object 84123 using sensors RME02, ABQ01 in TEST mode, schedule six revisits per hour, start a 12-hour timeline, classified as 'C', with a priority of 2. Starting at 2024-05-21 19:20:00.150000+00:00, ending at 2024-05-21 22:30:00.250000+00:00. Implement SIDEREAL tracking and LEO orbital regime, and set number of frames to 32 and integration time to 5 seconds.": "PeriodicRevisitObjective",
    "Track 43567 using sensors ABQ42, LMNT22 in TEST mode, schedule one revisit per hour, initiate with a 30-hour plan, marked as 'C', with priority set to 5. Begins at 2024-05-21 19:20:00.150000+00:00 and concludes at 2024-05-21 22:30:00.250000+00:00. Choose RATE_TRACK_SIDEREAL for tracking type and operate in MEO orbital regime, and set number of frames to 2 and integration time to 3 seconds.": "PeriodicRevisitObjective",
    # CatalogMaintenanceObjective
    "Make a new catalog maintenance for sensors RME02 and LMNT01 with U markings and TEST mode, priority 12, patience of 10 mins, ending after 25 mins. Start at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. Set RATE_TRACK_SIDEREAL for the tracking type and operate in LEO orbital regime.": "CatalogMaintenanceObjective",
    "Schedule a new catalog task for sensors ABQ04 and UKR05 with REAL mode, classification marking S, priority 8, patience of 25 minutes, and an end time offset of 35 minutes. Begin on 2024-05-21 19:20:00.150000+00:00 and finish by 2024-05-21 22:30:00.250000+00:00. Set RATE_TRACK for tracking. Orbital regime is GEO.": "CatalogMaintenanceObjective",
    "Configure catalog maintenance for sensors UKR07 and RME04 in TEST mode, with classification marking C, priority 12, patience of 15 minutes, concluding 40 minutes later. Starting at 2024-05-21 19:20:00.150000+00:00, ending at 2024-05-21 22:30:00.250000+00:00. Use SIDEREAL tracking. Set the orbital regime to MEO.": "CatalogMaintenanceObjective",
    "Set up a catalog entry for sensors RME12 and ABQ09, operating in REAL mode, with a U classification, a high priority of 18, 30 minutes of patience, and completion after 45 minutes. Starts at 2024-05-21 19:20:00.150000+00:00 and ends at 2024-05-21 22:30:00.250000+00:00. Set RATE_TRACK_SIDEREAL as the tracking type. Orbital regime is XGEO.": "CatalogMaintenanceObjective",
    "Create a maintenance task for sensors LMNT05 and LMNT06 in TEST mode, marked as TS, with a priority of 10, a 30-minute patience window, and an end time offset of 20 minutes. Start at 2024-05-21 19:20:00.150000+00:00, concluding at 2024-05-21 22:30:00.250000+00:00. Set RATE_TRACK for tracking. Orbital regime is LEO.": "CatalogMaintenanceObjective",
    "Plan a catalog operation for sensors LMNT11 and RME16 using TEST mode, S classification, with a priority of 13, patience for 20 minutes, and finishing after 30 minutes. Begins at 2024-05-21 19:20:00.150000+00:00 and ends at 2024-05-21 22:30:00.250000+00:00. Use RATE_TRACK_SIDEREAL. Operate in GEO orbital regime.": "CatalogMaintenanceObjective",
    "Create a new catalog maintenance for sensors RME15 and UKR03 with REAL mode, U//FOUO marking, priority 14, patience of 50 minutes, and set to end after 70 minutes. Begins at 2024-05-21 19:20:00.150000+00:00 and concludes at 2024-05-21 22:30:00.250000+00:00. Set SIDEREAL for tracking type. Orbital regime should be MEO.": "CatalogMaintenanceObjective",
    # SearchObjective
    "Create a new search objective for target 12345 using sensor UKR12 with S marking and REAL data mode, priority set to 5, and collect request type RATE_TRACK_SIDEREAL. Start the objective at 2024-05-22 09:30:00.000000+00:00 and end at 2024-05-22 11:00:00.000000+00:00. Set initial offset to 60 and final offset to 90, frame overlap percentage to 70%, number of frames to 8, integration time to 2 seconds, binning to 2, and end time offset to 45 minutes.": "SearchObjective",
    "Generate a search objective for target 98765 using sensor ABQ03 with TS marking and EXERCISE data mode, priority set to 3, and collect request type RATE_TRACK. Start time 2024-05-23 06:15:00.000000+00:00, end time 2024-05-23 08:45:00.000000+00:00. Initial offset 50, final offset 80, frame overlap 55%, 10 number of frames, 3 seconds integration time, binning 1, end time offset 30 minutes.": "SearchObjective",
    "Make a new search objective for target 54321 using sensor LMNT06 with U//FOUO marking and SIMULATED data mode, priority set to 7, and collect request type SIDEREAL. Start objective at 2024-05-24 00:00:00.000000+00:00 and end at 2024-05-24 02:30:00.000000+00:00. Initial offset 30, final offset 45, frame overlap percentage 65%, 6 number of frames, 2 seconds integration time, binning 3, end time offset 20 minutes.": "SearchObjective",
    "Create a search objective for target 11111 using sensor RME99 with U marking and TEST data mode, priority set to 10, and collect request type RATE_TRACK_SIDEREAL. Start time 2024-05-25 10:00:00.000000+00:00, end time 2024-05-25 12:15:00.000000+00:00. Initial offset 40, final offset 70, 60 percent frame overlap, 7 number of frames, 1 second integration time, default binning, 25 minutes end time offset.": "SearchObjective",
    "Generate a search objective for target 22222 using sensor ABQ77 with C marking and REAL data mode, priority set to 4, and collect request type RATE_TRACK. Start time 2024-05-26 15:30:00.000000+00:00, end time 2024-05-26 18:00:00.000000+00:00. 35 initial offset, 55 final offset, frame overlap of 50%, 9 number of frames, 4 seconds integration time, binning 2, 40 minutes end time offset.": "SearchObjective",
    "Make a new search objective for target 33333 using sensor LMNT22 with S marking and EXERCISE data mode, priority set to 8, and collect request type SIDEREAL. Start objective at 2024-05-27 06:00:00.000000+00:00 and end at 2024-05-27 09:15:00.000000+00:00. Initial offset 45, final offset 75, frame overlap percentage 55%, 8 number of frames, 3 seconds integration time, binning 1, end time offset 35 minutes.": "SearchObjective",
    "Create a search objective for target 44444 using sensor UKR55 with TS marking and SIMULATED data mode, priority set to 2, and collect request type RATE_TRACK_SIDEREAL. Start time 2024-05-28 12:00:00.000000+00:00, end time 2024-05-28 14:30:00.000000+00:00. 50 initial offset, 90 final offset, frame overlap of 65%, 10 number of frames, 2 seconds integration time, default binning, 30 minutes end time offset.": "SearchObjective",
    "Generate a search objective for target 55555 using sensor RME11 with U//FOUO marking and TEST data mode, priority set to 6, and collect request type RATE_TRACK. Start time 2024-05-29 18:45:00.000000+00:00, end time 2024-05-30 00:00:00.000000+00:00. 60 initial offset, 80 final offset, frame overlap of 70%, 7 number of frames, 1 second integration time, binning 3, 45 minutes end time offset.": "SearchObjective",
    # DataEnrichmentObjective
    "Create a data enrichment objective for targets 12345, 67890, and 54321 using sensors RME01, LMNT02, and ABQ03. Set the classification marking to 'U//FOUO', data mode to 'REAL', and collect request type to 'RATE_TRACK'. Observe a maximum of 8 RSOs with 10 revisits per hour, planning for 48 hours. Start the objective at 2024-06-01 08:00:00.000000+00:00 and end at 2024-06-03 08:00:00.000000+00:00. Set the priority to 15.": "DataEnrichmentObjective",
    "Generate a data enrichment for targets 98765 and 43210 using sensor UKR01 with 'TS' markings and 'EXERCISE' mode. Set the priority to 25 and use 'SIDEREAL' collect request type. Begin the objective at 2024-07-15 18:30:00.250000+00:00 and conclude at 2024-07-16 06:30:00.250000+00:00. Observe 5 RSOs with 15 revisits per hour, planning for 18 hours.": "DataEnrichmentObjective",
    "Prepare a data enrichment objective for targets 13579 and 24680 using sensors RME04 and LMNT05 with 'S' markings and 'TEST' mode. Set the priority to the default value and use 'RATE_TRACK_SIDEREAL' collect request type. Start the objective at 2024-08-10 09:45:00.500000+00:00 and end at 2024-08-11 21:45:00.500000+00:00. Observe the maximum number of RSOs with 8 revisits per hour, planning for 30 hours.": "DataEnrichmentObjective",
    "Create a data enrichment for targets 11111, 22222, 33333, and 44444 using sensors ABQ06, UKR07, and RME08 with 'C' markings and 'SIMULATED' mode. Set the priority to 18 and use 'RATE_TRACK' collect request type. Start the objective at 2024-09-05 12:00:00.750000+00:00 and end at 2024-09-06 00:00:00.750000+00:00. Observe 7 RSOs with 20 revisits per hour, planning for 16 hours.": "DataEnrichmentObjective",
    "Generate a data enrichment objective for target 55555 using sensor LMNT09 with 'U' markings and 'REAL' mode. Set the priority to 22 and use 'SIDEREAL' collect request type. Begin the objective at 2024-10-20 03:15:00.000000+00:00 and conclude at 2024-10-21 15:15:00.000000+00:00. Observe the default number of RSOs with 10 revisits per hour, planning for 40 hours.": "DataEnrichmentObjective",
    "Prepare a data enrichment for targets 66666 and 77777 using sensors RME10 and ABQ11 with 'U//FOUO' markings and 'EXERCISE' mode. Set the priority to 19 and use 'RATE_TRACK_SIDEREAL' collect request type. Start the objective at 2024-11-11 16:30:00.250000+00:00 and end at 2024-11-12 04:30:00.250000+00:00. Observe 4 RSOs with 18 revisits per hour, planning for 20 hours.": "DataEnrichmentObjective",
    "Create a data enrichment objective for targets 88888, 99999, and 00000 using sensors UKR12, LMNT13, and RME14 with 'TS' markings and 'TEST' mode. Set the priority to 23 and use 'RATE_TRACK' collect request type. Start the objective at 2024-12-01 06:45:00.500000+00:00 and end at 2024-12-02 18:45:00.500000+00:00. Observe 9 RSOs with 6 revisits per hour, planning for 42 hours.": "DataEnrichmentObjective",
    "Generate a data enrichment for target 12121 using sensor ABQ15 with 'S' markings and 'SIMULATED' mode. Set the priority to the default value and use 'SIDEREAL' collect request type. Begin the objective at 2025-01-15 20:00:00.750000+00:00 and conclude at 2025-01-16 08:00:00.750000+00:00. Observe the maximum number of RSOs with 14 revisits per hour, planning for 14 hours.": "DataEnrichmentObjective",
    # SpectralClearingObjective
    "Create a spectral clearing objective for targets 78901 and 23456 using sensors LMNT02 and UKR05 with S markings and REAL mode, priority set to 8. Start the objective at 2024-06-01 09:30:00.000000+00:00 and end at 2024-06-02 18:15:00.000000+00:00. Set the patience to 60 minutes, and run integration at 1.5 seconds per frame for 15 total frames per intent, using a binning of 2.": "SpectralClearingObjective",
    "Generate a new spectral clearing objective for target 13579 using sensor RME01 with U//FOUO markings and SIMULATED mode, priority set to 12. Start the objective at 2024-07-15 16:45:00.000000+00:00 and end at 2024-07-16 02:30:00.000000+00:00. Set the patience to 20 minutes, and run integration at 3 seconds per frame for 8 total frames per intent, using the default binning of 1.": "SpectralClearingObjective",
    "Initiate a spectral clearing objective for targets 98765 and 43210 using sensors ABQ02 and RME04 with C markings and TEST mode, priority set to 6. Start the objective at 2024-08-10 11:00:00.000000+00:00 and end at 2024-08-11 09:15:00.000000+00:00. Set the patience to 90 minutes, and run integration at 2.5 seconds per frame for 10 total frames per intent, using a binning of 4.": "SpectralClearingObjective",
    "Create a spectral clearing objective for target 56789 using sensors LMNT01 and UKR03 with U markings and REAL mode, priority set to 14. Start the objective at 2024-09-05 19:30:00.000000+00:00 and end at 2024-09-06 07:45:00.000000+00:00. Set the patience to 30 minutes, and run integration at 1.8 seconds per frame for 20 total frames per intent, using a binning of 3.": "SpectralClearingObjective",
    "Generate a new spectral clearing objective for targets 24680 and 13579 using sensors RME02 and ABQ03 with TS markings and EXERCISE mode, priority set to 9. Start the objective at 2024-10-20 08:15:00.000000+00:00 and end at 2024-10-21 17:30:00.000000+00:00. Set the patience to 75 minutes, and run integration at 2.2 seconds per frame for 18 total frames per intent, using the default binning of 1.": "SpectralClearingObjective",
    "Initiate a spectral clearing objective for target 97531 using sensor LMNT04 with C markings and SIMULATED mode, priority set to 11. Start the objective at 2024-11-12 14:00:00.000000+00:00 and end at 2024-11-13 03:45:00.000000+00:00. Set the patience to 50 minutes, and run integration at 1.2 seconds per frame for 25 total frames per intent, using a binning of 2.": "SpectralClearingObjective",
    "Create a spectral clearing objective for targets 86420 and 75319 using sensors UKR02 and RME05 with S markings and TEST mode, priority set to 7. Start the objective at 2024-12-08 10:45:00.000000+00:00 and end at 2024-12-09 22:30:00.000000+00:00. Set the patience to 40 minutes, and run integration at 2.8 seconds per frame for 14 total frames per intent, using a binning of 4.": "SpectralClearingObjective",
    "Generate a new spectral clearing objective for target 19753 using sensor ABQ04 with U//FOUO markings and REAL mode, priority set to 13. Start the objective at 2025-01-03 18:00:00.000000+00:00 and end at 2025-01-04 06:15:00.000000+00:00. Set the patience to 25 minutes, and run integration at 3.5 seconds per frame for 6 total frames per intent, using the default binning of 1.": "SpectralClearingObjective",
}


class ObjectiveBenchmark:
    def __init__(self, api_url: str = "http://localhost:8888"):
        self.api_url = api_url
        self.results = []
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def test_single_objective(self, query: str, expected: str) -> Dict:
        """Test a single objective query against the API with rate limiting."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                async with httpx.AsyncClient() as client:
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

                    # Rate limiting delay
                    await asyncio.sleep(RATE_LIMIT_DELAY)

                    return {
                        "query": query,
                        "expected": expected,
                        "predicted": predicted,
                        "execution_time": execution_time,
                        "correct": expected == predicted,
                    }
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                await asyncio.sleep(
                    RATE_LIMIT_DELAY
                )  # Still apply rate limiting on error
                return {
                    "query": query,
                    "expected": expected,
                    "predicted": "ERROR",
                    "execution_time": 0,
                    "correct": False,
                }

    async def run_benchmark(self):
        """Run benchmark on all test cases with proper rate limiting."""
        all_results = []

        # Process iterations sequentially to avoid overwhelming the API
        for iteration in range(N_ITERATIONS):
            logger.info(f"Starting iteration {iteration + 1}/{N_ITERATIONS}")

            # Create tasks for this iteration
            tasks = [
                self.test_single_objective(query, expected)
                for query, expected in OBJECTIVE_TEST_CASES.items()
            ]

            # Run tasks with controlled concurrency
            iteration_results = await asyncio.gather(*tasks)
            all_results.extend(iteration_results)

            # Add a longer delay between iterations
            if iteration < N_ITERATIONS - 1:
                await asyncio.sleep(RATE_LIMIT_DELAY * 2)

        self.results = all_results
        return self.results

    def print_results_analysis(self):
        """Print comprehensive analysis of benchmark results."""
        df = pd.DataFrame(self.results)

        # Calculate metrics
        total_tests = len(df)
        successful_tests = len(df[df["correct"]])
        accuracy = accuracy_score(df["expected"], df["predicted"])
        avg_execution_time = df["execution_time"].mean()

        # Print summary
        print("\n=== Benchmark Summary ===")
        print(f"Total Test Cases: {len(OBJECTIVE_TEST_CASES)}")
        print(f"Iterations per Test Case: {N_ITERATIONS}")
        print(f"Total Tests Run: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Average Execution Time: {avg_execution_time:.3f}s")

        # Per-class metrics
        report = classification_report(
            df["expected"],
            df["predicted"],
            zero_division=0,
            output_dict=True,
        )

        # Only show active classes
        actual_classes = set(df["expected"].unique())
        # predicted_classes = set(df["predicted"].unique()) - {"ERROR"}  # Exclude ERROR
        # active_classes = actual_classes.union(predicted_classes)

        print("\n=== Per-Class Performance ===")
        for class_name in actual_classes:
            class_df = df[df["expected"] == class_name]
            class_accuracy = (
                len(class_df[class_df["correct"]]) / len(class_df)
                if len(class_df) > 0
                else 0
            )

            # Calculate misclassifications
            incorrect_predictions = (
                class_df[~class_df["correct"]]["predicted"].value_counts().to_dict()
            )

            metrics = report.get(class_name, {})
            if metrics:
                print(f"\nClass: {class_name}")
                print(f"Accuracy: {class_accuracy:.2%}")
                print(f"Precision: {metrics.get('precision', 0):.2%}")
                print(f"Recall: {metrics.get('recall', 0):.2%}")
                print(f"F1-Score: {metrics.get('f1-score', 0):.2%}")
                print(f"Num Cases: {metrics.get('support', 0)}")

                if incorrect_predictions:
                    print("Misclassifications:")
                    for wrong_class, count in incorrect_predictions.items():
                        percentage = (count / len(class_df)) * 100
                        print(
                            f"  - {wrong_class}: {count} ({percentage:.1f}% of cases)"
                        )
                else:
                    print("Misclassifications: None")

            print("-" * 20)


async def main():
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
            await asyncio.sleep(5)  # Wait before retry
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Could not connect to API after {max_retries} attempts: {e}"
                )
                return
            await asyncio.sleep(5)  # Wait before retry

    # Run benchmark
    benchmark = ObjectiveBenchmark(api_url)
    await benchmark.run_benchmark()
    benchmark.print_results_analysis()


if __name__ == "__main__":
    asyncio.run(main())
