import asyncio
import logging
from typing import Dict

import httpx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_ITERATIONS = 5  # Num of times to run each test case
RATE_LIMIT_DELAY = 3.0  # Seconds between requests
MAX_CONCURRENT_REQUESTS = 1  # Maximum number of concurrent requests

OBJECTIVE_TEST_CASES = {
    # ----- PeriodicRevisitObjective (examples) -----
    (
        "Create a PeriodicRevisitObjective for targets 12225,68887 using sensors RME05,LMNT06. "
        "Set S marking, TEST mode, priority 2, patience minutes 30, ignore other objective submissions false. "
        "Start objective at 2024-06-21 19:20:00+00:00. Set optimal frames per hour 400, number of frames 5, integration time 2 seconds."
    ): "PeriodicRevisitObjective",
    # (
    #     "Track object 44248 with sensors RME01 and LMNT45, revisiting twice per hour for the next 36 hours using TEST mode, "
    #     "'S' markings, and priority 2. Begin at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Use RATE_TRACK_SIDEREAL as the collect request type, operate in LEO orbital regime, and set number of frames to 5 with 2 seconds integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Track celestial object 21212 with sensors RME33 and ABQ42 in REAL mode, revisiting four times per hour over a 48‐hour plan, "
    #     "marked as 'U//FOUO' with priority 1. Begin at 2024-05-21 19:20:00.150000+00:00 and finish by 2024-05-21 22:30:00.250000+00:00. "
    #     "Employ RATE_TRACK for tracking and operate in GSO orbital regime, setting number of frames to 10 with 1 second integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Track RSO object 43567 using sensors ABQ42 and UKR88 in TEST mode, scheduling four revisits per hour for a 36‐hour plan, "
    #     "marked as 'C' with priority 3. Start at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Use SIDEREAL tracking and operate in MEO orbital regime, with 12 frames and 4 seconds integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Monitor celestial object 20394 with sensors UKR88 and RME02 in REAL mode, configuring five revisits per hour within a 42‐hour schedule, "
    #     "classified as 'U//FOUO' with priority 3. Begin at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. "
    #     "Apply RATE_TRACK_SIDEREAL tracking and operate in HEO orbital regime, setting 3 frames with 10 seconds integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Observe object 31705 with sensors LMNT33 and RME99 in REAL mode, performing three revisits per hour in a 24‐hour strategy, "
    #     "marked as 'S' with priority 1. Begin at 2024-05-21 19:20:00.150000+00:00 and end by 2024-05-21 22:30:00.250000+00:00. "
    #     "Set collect request type to RATE_TRACK and operate in GSO orbital regime, with 16 frames and 20 seconds integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Follow object 84123 using sensors RME02 and ABQ01 in TEST mode, scheduling six revisits per hour over a 12‐hour timeline, "
    #     "classified as 'C' with priority 2. Start at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Implement SIDEREAL tracking in LEO orbital regime, setting 32 frames with 5 seconds integration."
    # ): "PeriodicRevisitObjective",
    # (
    #     "Track 43567 using sensors ABQ42 and LMNT22 in TEST mode, scheduling one revisit per hour for a 30‐hour plan, "
    #     "marked as 'C' with priority 5. Begin at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. "
    #     "Choose RATE_TRACK_SIDEREAL for tracking and operate in MEO orbital regime, with 2 frames and 3 seconds integration."
    # ): "PeriodicRevisitObjective",
    # ----- CatalogMaintenanceObjective (examples) -----
    (
        "Create a CatalogMaintenanceObjective for sensors RME04 and LMNT02 with U markings, "
        "TEST mode, priority 12, patience of 10 mins, end time offset of 25 mins, visibility check false. "
        "Start at 2024-05-21 19:20:00+00:00, end at 2024-05-21 22:30:00+00:00. Use RATE_TRACK_SIDEREAL tracking in LEO regime. "
        "RSO ID list includes '12445,67889'."
    ): "CatalogMaintenanceObjective",
    # (
    #     "Make a new catalog maintenance for sensors RME02 and LMNT01 with U markings in TEST mode, priority 12, and 10 minutes of patience with an end time offset of 25 minutes. "
    #     "Start at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. "
    #     "Set tracking to RATE_TRACK_SIDEREAL and operate in LEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Schedule a new catalog task for sensors ABQ04 and UKR05 in REAL mode with S classification, priority 8, and 25 minutes of patience with an end time offset of 35 minutes. "
    #     "Begin at 2024-05-21 19:20:00.150000+00:00 and finish by 2024-05-21 22:30:00.250000+00:00. "
    #     "Set tracking to RATE_TRACK and operate in GEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Configure catalog maintenance for sensors UKR07 and RME04 in TEST mode with C marking, priority 12, and 15 minutes of patience with an end offset of 40 minutes. "
    #     "Start at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Use SIDEREAL tracking and operate in MEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Set up a catalog entry for sensors RME12 and ABQ09 in REAL mode with U classification, priority 18, and 30 minutes of patience with an end offset of 45 minutes. "
    #     "Start at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Set tracking to RATE_TRACK_SIDEREAL and operate in XGEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Create a maintenance task for sensors LMNT05 and LMNT06 in TEST mode marked as TS, with priority 10, 30 minutes of patience, and an end time offset of 20 minutes. "
    #     "Begin at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. "
    #     "Set tracking to RATE_TRACK and operate in LEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Plan a catalog operation for sensors LMNT11 and RME16 using TEST mode with S classification, priority 13, and 20 minutes of patience with an end offset of 30 minutes. "
    #     "Begin at 2024-05-21 19:20:00.150000+00:00 and end at 2024-05-21 22:30:00.250000+00:00. "
    #     "Use RATE_TRACK_SIDEREAL for tracking and operate in GEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # (
    #     "Create a new catalog maintenance for sensors RME15 and UKR03 in REAL mode with U//FOUO marking, priority 14, and 50 minutes of patience with an end offset of 70 minutes. "
    #     "Start at 2024-05-21 19:20:00.150000+00:00 and conclude at 2024-05-21 22:30:00.250000+00:00. "
    #     "Use SIDEREAL tracking and operate in MEO orbital regime."
    # ): "CatalogMaintenanceObjective",
    # ----- SearchObjective (examples) -----
    (
        "Create a SearchObjective for target 12345 using sensor UKR12. Set S marking, REAL mode, priority 5, "
        "RATE_TRACK_SIDEREAL tracking. Start at 2024-07-24 09:30:00+00:00, end at 2024-07-24 11:00:00+00:00. "
        "Initial offset 60 seconds, final offset 90 seconds, frame overlap 70%, end time offset 45 minutes. "
        "Use search type ALONG_TRACK with search start time 15 minutes after objective start."
    ): "SearchObjective",
    # (
    #     "Create a new search objective for target 12345 using sensor UKR12 with S marking in REAL mode, priority 5, and collect request type RATE_TRACK_SIDEREAL. "
    #     "Start at 2024-05-22 09:30:00.000000+00:00 and end at 2024-05-22 11:00:00.000000+00:00. "
    #     "Set initial offset to 60, final offset to 90, frame overlap to 70%, 8 frames, 2 seconds integration, binning 2, and an end time offset of 45 minutes."
    # ): "SearchObjective",
    # (
    #     "Generate a search objective for target 98765 using sensor ABQ03 with TS marking in EXERCISE mode, priority 3, and collect request type RATE_TRACK. "
    #     "Start at 2024-05-23 06:15:00.000000+00:00 and end at 2024-05-23 08:45:00.000000+00:00. "
    #     "Set initial offset to 50, final offset to 80, frame overlap to 55%, 10 frames, 3 seconds integration, binning 1, and an end time offset of 30 minutes."
    # ): "SearchObjective",
    # (
    #     "Make a new search objective for target 54321 using sensor LMNT06 with U//FOUO marking in SIMULATED mode, priority 7, and collect request type SIDEREAL. "
    #     "Start at 2024-05-24 00:00:00.000000+00:00 and end at 2024-05-24 02:30:00.000000+00:00. "
    #     "Set initial offset to 30, final offset to 45, frame overlap to 65%, 6 frames, 2 seconds integration, binning 3, and an end time offset of 20 minutes."
    # ): "SearchObjective",
    # (
    #     "Create a search objective for target 11111 using sensor RME99 with U marking in TEST mode, priority 10, and collect request type RATE_TRACK_SIDEREAL. "
    #     "Start at 2024-05-25 10:00:00.000000+00:00 and end at 2024-05-25 12:15:00.000000+00:00. "
    #     "Set initial offset to 40, final offset to 70, frame overlap to 60%, 7 frames, 1 second integration, default binning, and an end time offset of 25 minutes."
    # ): "SearchObjective",
    # (
    #     "Generate a search objective for target 22222 using sensor ABQ77 with C marking in REAL mode, priority 4, and collect request type RATE_TRACK. "
    #     "Start at 2024-05-26 15:30:00.000000+00:00 and end at 2024-05-26 18:00:00.000000+00:00. "
    #     "Set initial offset to 35, final offset to 55, frame overlap to 50%, 9 frames, 4 seconds integration, binning 2, and an end time offset of 40 minutes."
    # ): "SearchObjective",
    # (
    #     "Make a new search objective for target 33333 using sensor LMNT22 with S marking in EXERCISE mode, priority 8, and collect request type SIDEREAL. "
    #     "Start at 2024-05-27 06:00:00.000000+00:00 and end at 2024-05-27 09:15:00.000000+00:00. "
    #     "Set initial offset to 45, final offset to 75, frame overlap to 55%, 8 frames, 3 seconds integration, binning 1, and an end time offset of 35 minutes."
    # ): "SearchObjective",
    # (
    #     "Create a search objective for target 44444 using sensor UKR55 with TS marking in SIMULATED mode, priority 2, and collect request type RATE_TRACK_SIDEREAL. "
    #     "Start at 2024-05-28 12:00:00.000000+00:00 and end at 2024-05-28 14:30:00.000000+00:00. "
    #     "Set initial offset to 50, final offset to 90, frame overlap to 65%, 10 frames, 2 seconds integration, default binning, and an end time offset of 30 minutes."
    # ): "SearchObjective",
    # (
    #     "Generate a search objective for target 55555 using sensor RME11 with U//FOUO marking in TEST mode, priority 6, and collect request type RATE_TRACK. "
    #     "Start at 2024-05-29 18:45:00.000000+00:00 and end at 2024-05-30 00:00:00.000000+00:00. "
    #     "Set initial offset to 60, final offset to 80, frame overlap to 70%, 7 frames, 1 second integration, binning 3, and an end time offset of 45 minutes."
    # ): "SearchObjective",
    # ----- DataEnrichmentObjective (examples) -----
    (
        "Create a DataEnrichmentObjective for targets 55441, 99886, 50051 using sensors RME31,LMNT34. "
        "Set U//FOUO marking, REAL mode, RATE_TRACK tracking, and set max RSO to observe as 8, 10 revisits per hour. "
        "Start at 2025-01-21 08:00:00+00:00. Set visibility check true."
    ): "DataEnrichmentObjective",
    # (
    #     "Create a data enrichment objective for targets 12345, 67890, and 54321 using sensors RME01, LMNT02, and ABQ03. "
    #     "Set classification marking to 'U//FOUO', data mode to REAL, and collect request type to RATE_TRACK. "
    #     "Observe up to 8 RSOs with 10 revisits per hour over a 48‐hour plan, and set priority to 15. "
    #     "Start at 2024-06-01 08:00:00.000000+00:00 and end at 2024-06-03 08:00:00.000000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Generate a data enrichment objective for targets 98765 and 43210 using sensor UKR01 with TS markings in EXERCISE mode, "
    #     "set priority to 25, and use SIDEREAL as the collect request type. "
    #     "Observe 5 RSOs with 15 revisits per hour over an 18‐hour plan. Begin at 2024-07-15 18:30:00.250000+00:00 and conclude at 2024-07-16 06:30:00.250000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Prepare a data enrichment objective for targets 13579 and 24680 using sensors RME04 and LMNT05 with S markings in TEST mode. "
    #     "Use RATE_TRACK_SIDEREAL as the collect request type with default priority, and observe the maximum number of RSOs with 8 revisits per hour over a 30‐hour plan. "
    #     "Start at 2024-08-10 09:45:00.500000+00:00 and end at 2024-08-11 21:45:00.500000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Create a data enrichment objective for targets 11111, 22222, 33333, and 44444 using sensors ABQ06, UKR07, and RME08 with C markings in SIMULATED mode. "
    #     "Set priority to 18 and use RATE_TRACK as the collect request type. "
    #     "Observe 7 RSOs with 20 revisits per hour over a 16‐hour plan. Start at 2024-09-05 12:00:00.750000+00:00 and end at 2024-09-06 00:00:00.750000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Generate a data enrichment objective for target 55555 using sensor LMNT09 with U markings in REAL mode. "
    #     "Set priority to 22 and use SIDEREAL as the collect request type. "
    #     "Observe the default number of RSOs with 10 revisits per hour over a 40‐hour plan. Begin at 2024-10-20 03:15:00.000000+00:00 and conclude at 2024-10-21 15:15:00.000000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Prepare a data enrichment objective for targets 66666 and 77777 using sensors RME10 and ABQ11 with U//FOUO markings in EXERCISE mode. "
    #     "Set priority to 19 and use RATE_TRACK_SIDEREAL as the collect request type. "
    #     "Observe 4 RSOs with 18 revisits per hour over a 20‐hour plan. Start at 2024-11-11 16:30:00.250000+00:00 and end at 2024-11-12 04:30:00.250000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Create a data enrichment objective for targets 88888, 99999, and 00000 using sensors UKR12, LMNT13, and RME14 with TS markings in TEST mode. "
    #     "Set priority to 23 and use RATE_TRACK as the collect request type. "
    #     "Observe 9 RSOs with 6 revisits per hour over a 42‐hour plan. Start at 2024-12-01 06:45:00.500000+00:00 and end at 2024-12-02 18:45:00.500000+00:00."
    # ): "DataEnrichmentObjective",
    # (
    #     "Generate a data enrichment objective for target 12121 using sensor ABQ15 with S markings in SIMULATED mode. "
    #     "Use SIDEREAL as the collect request type with default priority, and observe the maximum number of RSOs with 14 revisits per hour over a 14‐hour plan. "
    #     "Begin at 2025-01-15 20:00:00.750000+00:00 and conclude at 2025-01-16 08:00:00.750000+00:00."
    # ): "DataEnrichmentObjective",
    # ----- GeodssRevisitObjective (examples) -----
    (
        "Create a GeodssRevisitObjective for targets 12345,67890 using sensors RME15,LMNT17. Set U//FOUO marking, REAL mode, "
        "priority 10, RATE_TRACK_SIDEREAL tracking. Start at 2024-08-11 19:20:00+00:00. Set readout_rate 1 (2MHz), gain_setting 0 (High Gain), "
        "soi_filter 1 (1% Light), auto_track_type 1 (Automatic), camera_mode 0 (Normal), array_kind 0 (Main), binning_mode 1 (HW Binning), "
        "scan_mode 1 (Single Frame)."
    ): "GeodssRevisitObjective",
    # (
    #     "Create a GeodssRevisitObjective for targets 11111,22222 using sensors RME50 and LMNT55. Set U//FOUO marking, REAL mode, "
    #     "priority 12, and use RATE_TRACK_SIDEREAL tracking. Start at 2024-07-20 20:00:00+00:00. "
    #     "Set readout_rate to 1, gain_setting to 0, soi_filter to 1, auto_track_type to 1, camera_mode to 0, "
    #     "array_kind to 0, binning_mode to 1, and scan_mode to 1."
    # ): "GeodssRevisitObjective",
    # (
    #     "Generate a GeodssRevisitObjective for targets 33333,44444 using sensors ABQ33 and UKR66. Set U//FOUO marking in REAL mode, "
    #     "priority 9, with RATE_TRACK_SIDEREAL tracking. Start at 2024-08-05 19:30:00+00:00. "
    #     "Configure readout_rate 1, gain_setting 1, soi_filter 0, auto_track_type 2, camera_mode 1, array_kind 1, binning_mode 1, and scan_mode 1."
    # ): "GeodssRevisitObjective",
    # (
    #     "Plan a GeodssRevisitObjective for targets 55555,66666 with sensors LMNT60 and RME65. Set U//FOUO marking, REAL mode, "
    #     "priority 8, and RATE_TRACK_SIDEREAL tracking. Begin at 2024-09-15 18:45:00+00:00. "
    #     "Set readout_rate 1, gain_setting 0, soi_filter 1, auto_track_type 1, camera_mode 0, array_kind 0, binning_mode 1, and scan_mode 1."
    # ): "GeodssRevisitObjective",
    # (
    #     "Develop a GeodssRevisitObjective for targets 77777,88888 using sensors ABQ44 and LMNT70. Set U//FOUO marking, REAL mode, "
    #     "priority 10, and use RATE_TRACK_SIDEREAL tracking. Start at 2024-10-10 21:00:00+00:00. "
    #     "Configure readout_rate as 1, gain_setting as 0, soi_filter as 1, auto_track_type as 2, camera_mode as 0, array_kind as 0, "
    #     "binning_mode as 1, and scan_mode as 1."
    # ): "GeodssRevisitObjective",
    # (
    #     "Create a GeodssRevisitObjective for targets 99999,00000 using sensors UKR77 and RME80. Set U//FOUO marking, REAL mode, "
    #     "priority 11, and use RATE_TRACK_SIDEREAL tracking. Start at 2024-11-25 19:20:00+00:00. "
    #     "Set readout_rate 1, gain_setting 0, soi_filter 1, auto_track_type 1, camera_mode 0, array_kind 0, binning_mode 1, and scan_mode 1."
    # ): "GeodssRevisitObjective",
    # ----- SensorCheckoutObjective (examples) -----
    (
        "Create a SensorCheckoutObjective with classification_marking='U' and sensor_name='RME01'. "
        "Set data_mode='REAL', orbital_regime='GEO', collect_request_type='RATE_TRACK_SIDEREAL', priority=10, "
        "revisits_per_hour=1.0, objective_start_time='2025-02-01 19:20:00+00:00', visibility_check=true, "
        "patience_minutes=30, number_of_frames=5, integration_time=2."
    ): "SensorCheckoutObjective",
    # (
    #     "Create a SensorCheckoutObjective with classification marking U for sensor RME02. Use REAL mode, GEO regime, "
    #     "and RATE_TRACK_SIDEREAL as the collect request type, with priority 8. Set revisits per hour to 1.0, 30 mins patience, "
    #     "5 frames, and 2 sec integration. Start at 2025-03-01 12:00:00+00:00."
    # ): "SensorCheckoutObjective",
    # (
    #     "Generate a SensorCheckoutObjective for sensor ABQ11 with U marking in REAL mode and GEO regime, using RATE_TRACK_SIDEREAL. "
    #     "Set priority 10, revisits per hour 1.0, 25 mins patience, 5 frames, and 2 sec integration. Start at 2025-03-05 14:30:00+00:00."
    # ): "SensorCheckoutObjective",
    # (
    #     "Plan a SensorCheckoutObjective for sensor LMNT07 with U marking, REAL mode, GEO regime, and RATE_TRACK_SIDEREAL tracking. "
    #     "Set priority 9, revisits per hour 1.0, 30 mins patience, 4 frames, and 3 sec integration. Begin at 2025-03-10 09:15:00+00:00."
    # ): "SensorCheckoutObjective",
    # (
    #     "Develop a SensorCheckoutObjective for sensor UKR09 with U marking in REAL mode and GEO orbital regime, using RATE_TRACK_SIDEREAL. "
    #     "Set priority 7, revisits per hour 1.0, 30 mins patience, 6 frames, and 2 sec integration. Start at 2025-03-15 16:45:00+00:00."
    # ): "SensorCheckoutObjective",
    # (
    #     "Create a SensorCheckoutObjective with classification U for sensor RME08 in REAL mode with GEO regime, "
    #     "using RATE_TRACK_SIDEREAL tracking, priority 10, revisits per hour 1.0, 35 mins patience, 5 frames, and 2 sec integration. "
    #     "Start at 2025-03-20 11:00:00+00:00."
    # ): "SensorCheckoutObjective",
    # ----- SingleIntentObjective (examples) -----
    (
        "Create a SingleIntentObjective with target ID 11223, RSO ID 66778, using sensors RME22,LMNT24. "
        "Set U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, priority 10. "
        "Start objective at 2024-10-01 11:20:00+00:00. Set number of frames to 5, integration time 2 seconds, binning 2."
    ): "SingleIntentObjective",
    # (
    #     "Create a SingleIntentObjective with target ID 101010 and RSO ID 202020 using sensors RME12 and LMNT14. "
    #     "Set U marking, REAL mode, and RATE_TRACK_SIDEREAL tracking, with priority 8. Start at 2025-04-01 10:00:00+00:00. "
    #     "Set number of frames to 5, integration time to 2 sec, and binning 2."
    # ): "SingleIntentObjective",
    # (
    #     "Generate a SingleIntentObjective for target ID 303030 and RSO ID 404040 using sensors ABQ20 and UKR15. "
    #     "Use U marking in REAL mode with RATE_TRACK_SIDEREAL, priority 7. Start at 2025-04-05 11:30:00+00:00, with 6 frames, 2 sec integration, and binning 1."
    # ): "SingleIntentObjective",
    # (
    #     "Plan a SingleIntentObjective with target ID 505050 and RSO ID 606060 using sensors LMNT16 and RME18. "
    #     "Set U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, and priority 9. Start at 2025-04-10 09:45:00+00:00, with 5 frames, 3 sec integration, and binning 2."
    # ): "SingleIntentObjective",
    # (
    #     "Develop a SingleIntentObjective for target ID 707070 and RSO ID 808080 using sensors ABQ30 and UKR25. "
    #     "Set U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, priority 10, and start at 2025-04-15 14:00:00+00:00. "
    #     "Set number of frames to 7, integration time to 2 sec, and binning 3."
    # ): "SingleIntentObjective",
    # (
    #     "Create a SingleIntentObjective with target ID 909090 and RSO ID 101010 using sensors RME22 and LMNT24. "
    #     "Use U marking, REAL mode, RATE_TRACK_SIDEREAL tracking, and priority 8. Start at 2025-04-20 12:30:00+00:00, "
    #     "with 5 frames, 2 sec integration, and binning 2."
    # ): "SingleIntentObjective",
    # ----- UctObservationObjective (examples) -----
    (
        "Create a UctObservationObjective for UCT RSOs 12345,67890 using sensors RME18, LMNT19. "
        "Set U//FOUO marking, REAL mode, GEO regime, priority 10, 6 revisits per hour. "
        "Start at 2024-09-12 19:20:00+00:00. Enable sorting by brightest UCT, set end time offset to 60 minutes, visibility check true. "
        "Set number of frames to 5 and integration time 2 seconds."
    ): "UctObservationObjective",
    # (
    #     "Create a UctObservationObjective for UCT RSOs 11111,22222 using sensors RME30 and LMNT35. "
    #     "Set U//FOUO marking, REAL mode with GEO regime, priority 9, and 6 revisits per hour. "
    #     "Start at 2025-05-01 18:00:00+00:00. Enable sorting by brightest UCT, set end time offset to 60 mins, with 5 frames and 2 sec integration."
    # ): "UctObservationObjective",
    # (
    #     "Generate a UctObservationObjective for UCT RSOs 33333,44444 using sensors ABQ40 and UKR30. "
    #     "Set U//FOUO marking, REAL mode with GEO regime, priority 10, and 6 revisits per hour. "
    #     "Start at 2025-05-05 17:30:00+00:00, enable sorting by brightest UCT, with end time offset 60 mins, 5 frames, and 2 sec integration."
    # ): "UctObservationObjective",
    # (
    #     "Plan a UctObservationObjective for UCT RSOs 55555,66666 using sensors RME35 and LMNT38. "
    #     "Use U//FOUO marking in REAL mode with GEO regime, priority 8, and 6 revisits per hour. "
    #     "Begin at 2025-05-10 19:15:00+00:00, enable sorting by brightest UCT, set end time offset to 60 mins, 5 frames, and 2 sec integration."
    # ): "UctObservationObjective",
    # (
    #     "Design a UctObservationObjective for UCT RSOs 77777,88888 using sensors ABQ50 and UKR45. "
    #     "Set U//FOUO marking, REAL mode with GEO regime, priority 7, and 6 revisits per hour. "
    #     "Start at 2025-05-15 20:00:00+00:00, enable sorting by brightest UCT, with an end time offset of 60 mins, 5 frames, and 2 sec integration."
    # ): "UctObservationObjective",
    # (
    #     "Create a UctObservationObjective for UCT RSOs 99999,00000 using sensors RME40 and LMNT42. "
    #     "Set U//FOUO marking, REAL mode with GEO regime, priority 10, and 6 revisits per hour. "
    #     "Start at 2025-05-20 18:45:00+00:00, enable sorting by brightest UCT, set end time offset to 60 mins, with 5 frames and 2 sec integration."
    # ): "UctObservationObjective",
    # ----- BaselineAutonomyObjective (examples) -----
    (
        "Create a BaselineAutonomyObjective with UUID '123e4567-e89b-12d3-a456-426614174000'. "
        "Use U markings, REAL mode, LIGHT frame type, priority 1000. "
        "Use following RSO ids: 11112, 99996, and 59591 along with catalog IDs: 17180 and 19210, with no end time for continuous running."
    ): "BaselineAutonomyObjective",
    # (
    #     "Create a BaselineAutonomyObjective with UUID 'abc123-uuid-001'. Use U markings in REAL mode, LIGHT frame type, "
    #     "priority 1000. Specify baseline autonomy RSO as catalog IDs '10001,10002' and include RSO IDs 55555 and 66666."
    # ): "BaselineAutonomyObjective",
    # (
    #     "Generate a BaselineAutonomyObjective with UUID 'def456-uuid-002'. Use U markings, REAL mode, LIGHT frame type, "
    #     "and priority 1000. Set baseline autonomy RSO to '20001,20002' and provide RSO IDs 77777 and 88888."
    # ): "BaselineAutonomyObjective",
    # (
    #     "Plan a BaselineAutonomyObjective with UUID 'ghi789-uuid-003'. Use U markings in REAL mode, LIGHT frame type, "
    #     "priority 1000. Specify baseline autonomy RSO as '30001,30002' and include RSO IDs 99999 and 11111, with no defined end time."
    # ): "BaselineAutonomyObjective",
    # (
    #     "Design a BaselineAutonomyObjective with UUID 'jkl012-uuid-004'. Use U markings, REAL mode, LIGHT frame type, "
    #     "and priority 1000. Set baseline autonomy RSO as '40001,40002' and include RSO IDs 22222 and 33333."
    # ): "BaselineAutonomyObjective",
    # (
    #     "Create a BaselineAutonomyObjective with UUID 'mno345-uuid-005'. Use U markings in REAL mode, LIGHT frame type, "
    #     "priority 1000. Specify baseline autonomy RSO as '50001,50002' and provide RSO IDs 44444 and 55555."
    # ): "BaselineAutonomyObjective",
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
