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
    # New Example 10: CatalogMaintenanceObjective
    (
        "Initiate a CatalogMaintenanceObjective. Sensors involved are 'XYZ78' and 'ABC01'. Use 'C' classification. "
        "Mode is 'SIMULATED'. Priority level: 50. Set patience to 15 minutes and end time offset to 30 minutes. "
        "Start the objective on 2025-10-10 at 10:00:00 UTC and end it on 2025-10-10 at 15:30:00 UTC. Orbital regime is MEO. "
        "Tracking type is RATE_TRACK. Visibility check should be enabled."
    ): {
        "classification_marking": "C",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["XYZ78", "ABC01"],
        "rso_id_list": [],
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 10, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 11: SearchObjective
    (
        "Configure a SearchObjective for target ID 'TGT007' with sensor 'SNS_ALPHA'. Search pattern: 'CROSS_TRACK'. "
        "Classification: 'TS', Data Mode: 'EXERCISE'. Set initial offset to 45s, final offset to 75s, frame overlap to 0.6. Priority: 3. "
        "Start time: 2025-11-15 08:00:00+00:00, end time: 2025-11-15 10:00:00+00:00. Integration time: 1.5s, number of frames: 10. Binning: 2."
    ): {
        "classification_marking": "TS",
        "target_id": "TGT007",
        "sensor_name": "SNS_ALPHA",
        "search_type": "CROSS_TRACK",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 20,
        "priority": 3,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 15, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 15, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 1.5,
        "search_start_time": "datetime.datetime(2025, 11, 15, 8, 15, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 12: GeodssRevisitObjective
    (
        "Establish a GeodssRevisitObjective. Targets: 'SAT44', 'SAT55'. Sensors: 'GEO_A', 'GEO_B'. "
        "Marking: 'U//FOUO'. Priority: 7. Start: 2025-12-01 00:00:00 UTC. Readout: 1MHz (0). Gain: Low (1). "
        "SOI Filter: 10% Light (2). Auto Track: Manual (2). Camera Mode: Zoomed EBS (1). Array: Photometer (1). "
        "Binning: No Binning (0). Scan: Continuous (0). Patience: 45 mins. Number of observations: 3."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["SAT44", "SAT55"],
        "sensor_name_list": ["GEO_A", "GEO_B"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 45,
        "revisits_per_hour": None,
        "number_of_frames": None,
        "integration_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "acquisition_type": 0,
        "auto_track_type": 2,
        "auto_track_roi_position": 0,
        "camera_mode": 1,
        "observation_interval": 0.0,
        "num_observations": 3,
        "num_skip_frames": 0,
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "rate_track_verify": 0,
        "soi_filter_position": 2,
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "overscan": 0,
        "command": 0,
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 13: PeriodicRevisitObjective
    (
        "Define a PeriodicRevisitObjective. Target IDs are 'P_RSO_1', 'P_RSO_2', 'P_RSO_3'. Sensors: 'SEN_X', 'SEN_Y'. "
        "Classification marking 'S'. Data mode is 'TEST'. Collect type: 'SIDEREAL'. Visibility check: true. Patience: 20 mins. "
        "Revisits per hour: 4.5. Number of frames: 3. Integration time: 0.5 seconds. "
        "Objective start: 2026-01-15 12:00:00+00:00. Objective end: 2026-01-16 12:00:00+00:00. Priority 5. "
        "Ignore other submissions: true. Optimal frames/hr: 300. Intent start: 2026-01-15 13:00:00+00:00. "
        "Intent end: 2026-01-16 11:00:00+00:00."
    ): {
        "classification_marking": "S",
        "target_id_list": ["P_RSO_1", "P_RSO_2", "P_RSO_3"],
        "sensor_name_list": ["SEN_X", "SEN_Y"],
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 4.5,
        "number_of_frames": 3,
        "integration_time": 0.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 16, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": "datetime.datetime(2026, 1, 15, 13, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 1, 16, 11, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 14: UctObservationObjective
    (
        "Task a UctObservationObjective for UCT RSOs 'UCT_A1', 'UCT_B2'. Sensors: 'UCT_SEN1'. "
        "Classification: 'U'. Data Mode 'REAL'. Orbital Regime: 'LEO'. Visibility: false. Patience: 25 mins. "
        "Revisits: 5.0 per hour. Frames: 2. Integration: 3.0s. Binning: 4. End time offset: 40 mins. "
        "Start time: 2026-02-20 18:30:00+00:00. Priority: 15. Sort by brightest: false."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["UCT_A1", "UCT_B2"],
        "sensor_name_list": ["UCT_SEN1"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "LEO",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 5.0,
        "number_of_frames": 2,
        "integration_time": 3.0,
        "binning": 4,
        "end_time_offset_minutes": 40,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 18, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 15: SingleIntentObjective
    (
        "Schedule a SingleIntentObjective. Target is 'TGT_SINGLE_001', RSO ID is 'RSO_SINGLE_XYZ'. "
        "Sensors: 'MAIN_CAM', 'AUX_CAM'. Classification: 'C'. Data mode: 'SIMULATED'. Tracking: 'RATE_TRACK'. "
        "Frames: 1. Integration: 10.0s. Priority: 3. Binning: 1. "
        "Objective Start: 2026-03-10 05:00:00+00:00. Intent Start: 2026-03-10 05:05:00+00:00. "
        "Intent End: 2026-03-10 05:10:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id": "TGT_SINGLE_001",
        "rso_id": "RSO_SINGLE_XYZ",
        "sensor_name_list": ["MAIN_CAM", "AUX_CAM"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 1,
        "integration_time": 10.0,
        "priority": 3,
        "binning": 1,
        "intent_start_time": "datetime.datetime(2026, 3, 10, 5, 5, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 3, 10, 5, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 16: DataEnrichmentObjective
    (
        "Plan a DataEnrichmentObjective. Targets: 'DE_T1', 'DE_T2'. Sensors: 'ENRICH_S1'. "
        "Classification: 'TS'. Data Mode: 'EXERCISE'. Collect Type: 'SIDEREAL'. Max RSOs: 5. Revisits/hr: 8.0. "
        "Start: 2026-04-05 02:00:00+00:00. End: 2026-04-05 12:00:00+00:00. Priority: 25. Visibility Check: false. Binning: 2."
    ): {
        "classification_marking": "TS",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["DE_T1", "DE_T2"],
        "sensor_name_list": ["ENRICH_S1"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": 2,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 2, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 4, 5, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 17: SensorCheckoutObjective
    (
        "Set up a SensorCheckoutObjective for sensor 'CHK_SENSOR_7'. Classification: 'U//FOUO'. "
        "Orbital Regime: 'XGEO'. Data mode: 'TEST'. Tracking: 'RATE_TRACK'. Visibility: false. Patience: 40 mins. "
        "Revisits: 0.5 per hour. Frames: 7. Integration: 2.5s. Binning: 1. "
        "Start: 2026-05-01 00:00:00+00:00. End: 2026-05-01 06:00:00+00:00. Priority: 8."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "CHK_SENSOR_7",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 40,
        "revisits_per_hour": 0.5,
        "number_of_frames": 7,
        "integration_time": 2.5,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 5, 1, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 8,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 18: BaselineAutonomyObjective
    (
        "Deploy BaselineAutonomyObjective. UUID is 'fedcba98-7654-3210-fedc-ba9876543210'. "
        "Marking: 'S'. Data Mode: 'SIMULATED'. Frame Type: 'DARK'. Priority: 750. "
        "Baseline RSOs (Catalog IDs): 'CAT123,CAT456,CAT789'. RSO ID list to populate: 'RSO_X1','RSO_Y2'. "
        "Objective End Time: 2026-06-30 23:59:59+00:00."
    ): {
        "objective_uuid": "fedcba98-7654-3210-fedc-ba9876543210",
        "classification_marking": "S",
        "data_mode": "SIMULATED",
        "frame_type": "DARK",
        "priority": 750,
        "baseline_autonomy_rso": "CAT123,CAT456,CAT789",
        "objective_end_time": "datetime.datetime(2026, 6, 30, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["RSO_X1", "RSO_Y2"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 19: CatalogMaintenanceObjective
    (
        "System, please initiate a CatalogMaintenanceObjective for the MEO orbital regime. Target sensors RME07 and LMNT09. "
        "The classification marking should be C. Use REAL data mode. Set the priority to 500. "
        "The patience period will be 15 minutes, and the end time offset is 30 minutes. Visibility check should be enabled. "
        "The objective is scheduled to start on 2025-06-10 at 10:00:00 UTC and conclude on 2025-06-10 at 14:00:00 UTC. "
        "Include RSO IDs '22334' and '88776'."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 500,
        "rso_id_list": ["22334", "88776"],
        "sensor_name_list": ["RME07", "LMNT09"],
        "visibility_check": True,
    },
    # New Example 20: SearchObjective
    (
        "Generate a SearchObjective to locate target ID 77889 using sensor ZYX01. "
        "The operation requires a TS classification marking and will operate in SIMULATED mode. Employ SIDEREAL tracking. "
        "The objective will commence at 2025-08-15 05:00:00 UTC and terminate at 2025-08-15 07:30:00 UTC. "
        "Set an initial offset of 45 seconds and a final offset of 75 seconds. "
        "The frame overlap should be 60%, and the end time offset is 35 minutes. "
        "The search type is CROSS_TRACK, and please specify the search start time as 20 minutes after the objective begins. Priority is 3."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 8, 15, 7, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 15, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 8, 15, 5, 20, 0, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "ZYX01",
        "target_id": "77889",
        "visibility_check": False,
    },
    # New Example 21: GeodssRevisitObjective
    (
        "Configure a GeodssRevisitObjective for target IDs 'TGT001' and 'TGT002', utilizing sensors GEO01 and GEO02. "
        "This objective has a U//FOUO classification, operates in REAL mode with RATE_TRACK tracking. The priority is set to 15. "
        "Objective commencement is 2025-09-20 12:00:00 UTC. "
        "Please use a readout rate of 0 (1MHz), gain setting of 1 (Low Gain), SOI filter 2 (10% Light), "
        "automatic auto track type, normal camera mode, main array kind, no binning mode, and continuous scan mode."
    ): {
        "acquisition_type": 0,
        "array_kind": 0,
        "auto_track_roi_position": 0,
        "auto_track_type": 1,
        "binning_mode": 0,
        "camera_mode": 0,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,
        "num_skip_frames": 0,
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 20, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 30,
        "priority": 15,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEO01", "GEO02"],
        "soi_filter_position": 2,
        "target_id_list": ["TGT001", "TGT002"],
        "visibility_check": False,
    },
    # New Example 22: PeriodicRevisitObjective
    (
        "Establish a PeriodicRevisitObjective. The targets are 'PRTARG01', 'PRTARG02', and 'PRTARG03', "
        "assigned to sensors PSENS01 and PSENS02. It's marked as C, runs in TEST mode, with a priority of 7. "
        "Patience is 25 minutes. Let's ignore other objective submissions. "
        "The objective starts on 2025-10-05 08:30:00+00:00. "
        "We need an optimal 300 frames per hour, 3 frames per intent, and an integration time of 1.5 seconds. "
        "SIDEREAL tracking will be used."
    ): {
        "classification_marking": "C",
        "target_id_list": ["PRTARG01", "PRTARG02", "PRTARG03"],
        "sensor_name_list": ["PSENS01", "PSENS02"],
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": None,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 10, 5, 8, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 23: UctObservationObjective
    (
        "Initiate a UctObservationObjective for UCT RSO IDs 'UCT111' and 'UCT222'. "
        "Assign this task to sensors UCTSENS01 and UCTSENS02. The classification is S, data mode is EXERCISE, "
        "and it's for the LEO orbital regime. Set priority to 12. Require 4 revisits per hour. "
        "The objective should start on 2025-11-15 22:00:00+00:00. Disable sorting by brightest UCT. "
        "The end time offset is 40 minutes. Visibility check should be active. "
        "We need 4 frames and an integration time of 2.5 seconds. RATE_TRACK_SIDEREAL is the tracking type."
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["UCT111", "UCT222"],
        "sensor_name_list": ["UCTSENS01", "UCTSENS02"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "LEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 4.0,
        "number_of_frames": 4,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 40,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 15, 22, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 12,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 24: SingleIntentObjective
    (
        "Please create a SingleIntentObjective. The target ID is 'TGTX05', and the RSO ID is 'RSOY06'. "
        "This will use sensors 'SNSRALPHA' and 'SNSRBETA'. Mark this as U//FOUO. "
        "The data mode is SIMULATED, and tracking is RATE_TRACK. Set the priority to 8. "
        "The objective starts on 2025-12-01 03:15:00+00:00. "
        "Capture 6 frames with an integration time of 3 seconds and binning set to 4."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "TGTX05",
        "rso_id": "RSOY06",
        "sensor_name_list": ["SNSRALPHA", "SNSRBETA"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 6,
        "integration_time": 3.0,
        "priority": 8,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 3, 15, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 25: DataEnrichmentObjective
    (
        "We need a DataEnrichmentObjective for the following target IDs: 'ENRICH01', 'ENRICH02'. "
        "This will be handled by sensors 'ENRSENS01' and 'ENRSENS02'. The classification is 'TS', data mode 'REAL'. "
        "Employ 'SIDEREAL' tracking. Set the maximum RSOs to observe to 10, and require 15 revisits per hour. "
        "The objective starts 2026-01-10 18:00:00+00:00. Ensure visibility check is disabled. The priority is 25."
    ): {
        "classification_marking": "TS",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["ENRICH01", "ENRICH02"],
        "sensor_name_list": ["ENRSENS01", "ENRSENS02"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 26: SensorCheckoutObjective
    (
        "Let's define a SensorCheckoutObjective with a 'C' classification marking for sensor 'CHKOUT01'. "
        "The data mode is 'TEST', and the orbital regime is 'MEO'. Use 'RATE_TRACK' for collection. Priority is 5. "
        "Set revisits per hour to 0.5. The objective will begin on 2026-02-20 09:00:00 UTC. "
        "Visibility check is false. Patience is 45 minutes. Specify 3 frames and an integration time of 1 second. "
        "Binning should be 1."
    ): {
        "classification_marking": "C",
        "sensor_name": "CHKOUT01",
        "orbital_regime": "MEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 45,
        "revisits_per_hour": 0.5,
        "number_of_frames": 3,
        "integration_time": 1.0,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 27: BaselineAutonomyObjective
    (
        "Configure a BaselineAutonomyObjective with the specific UUID 'abcdef01-2345-6789-abcd-ef0123456789'. "
        "This objective will have TS marking, operate in EXERCISE data mode, using DARK frames. The priority is set to 1500. "
        "For tracking, include catalog IDs 'CAT777' and 'CAT888'. "
        "The RSO ID list for this operation includes 'RSOBASE01' and 'RSOBASE02'. "
        "This objective has an end time of 2026-03-30 23:59:59 UTC."
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "TS",
        "data_mode": "EXERCISE",
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "CAT777,CAT888",
        "objective_end_time": "datetime.datetime(2026, 3, 30, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["RSOBASE01", "RSOBASE02"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # 28. CatalogMaintenanceObjective
    (
        "Deploy a CatalogMaintenanceObjective utilizing sensors LMNT23 and RME12 for MEO regime with TS classification. "
        "Configure for SIMULATED mode with priority 8, patience window of 15 mins, and end offset of 35 mins. "
        "Start operations on 2025-06-15 06:45:00+00:00 and conclude by 2025-06-15 09:30:00+00:00. "
        "Apply RATE_TRACK tracking for RSO identifiers 55123 and 78901. Enable visibility verification."
    ): {
        "classification_marking": "TS",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 35,
        "priority": 8,
        "sensor_name_list": ["LMNT23", "RME12"],
        "rso_id_list": ["55123", "78901"],
        "objective_start_time": "datetime.datetime(2025, 6, 15, 6, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 9, 30, tzinfo=TzInfo(UTC))",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
        "objective_uuid": None,
        "binning": None,
    },
    # 29. SearchObjective
    (
        "Initiate a SearchObjective for target 98765 with sensor LMNT08 in CROSS_TRACK mode. "
        "Apply C marking in TEST environment with priority 3. Schedule from 2025-08-30 14:00:00+00:00 to 2025-08-30 16:30:00+00:00. "
        "Set initial search offset at 45 seconds, final offset at 75 seconds with 60% frame overlap. "
        "Configure end time buffer of 30 minutes and begin search operations 20 minutes after objective start."
    ): {
        "classification_marking": "C",
        "target_id": "98765",
        "sensor_name": "LMNT08",
        "search_type": "CROSS_TRACK",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 30,
        "priority": 3,
        "objective_start_time": "datetime.datetime(2025, 8, 30, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 30, 16, 30, tzinfo=TzInfo(UTC))",
        "search_start_time": "datetime.datetime(2025, 8, 30, 14, 20, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_uuid": None,
        "binning": None,
        "number_of_frames": None,
        "integration_time": None,
    },
    # 30. GeodssRevisitObjective
    (
        "Establish a GeodssRevisitObjective for monitoring RSOs 33333 and 44444 via sensors LMNT28 and RME33. "
        "Use S//FOUO marking in EXERCISE mode with priority 7. Begin at 2025-07-18 22:15:00+00:00. "
        "Configure instrumentation with readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), "
        "auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), "
        "binning_mode 0 (No Binning), scan_mode 0 (Continuous)."
    ): {
        "classification_marking": "S//FOUO",
        "target_id_list": ["33333", "44444"],
        "sensor_name_list": ["LMNT28", "RME33"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "priority": 7,
        "objective_start_time": "datetime.datetime(2025, 7, 18, 22, 15, tzinfo=TzInfo(UTC))",
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "soi_filter_position": 2,
        "auto_track_type": 2,
        "camera_mode": 1,
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "objective_name": "GeodssRevisitObjective",
        "objective_uuid": None,
        "objective_end_time": None,
        "visibility_check": False,
        "patience_minutes": 30,
    },
    # 31. PeriodicRevisitObjective
    (
        "Launch a PeriodicRevisitObjective for RSOs 77711 and 88855 with LMNT09 and RME27 sensors. "
        "Implement U//FOUO classification in REAL operational mode. Configure for 2.5 revisits per hour, "
        "capture 8 frames per observation with 3.5 second integration time. Start on 2025-09-10 03:20:00+00:00 "
        "with patience threshold of 25 minutes and priority level 4."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["77711", "88855"],
        "sensor_name_list": ["LMNT09", "RME27"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 2.5,
        "number_of_frames": 8,
        "integration_time": 3.5,
        "objective_start_time": "datetime.datetime(2025, 9, 10, 3, 20, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "objective_name": "PeriodicRevisitObjective",
        "objective_uuid": None,
        "objective_end_time": None,
        "binning": None,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "intent_start_time": None,
        "intent_end_time": None,
    },
    # 32. UctObservationObjective
    (
        "Instantiate a UctObservationObjective targeting UCT objects 22334, 55112, and 78998 using facilities RME25 and LMNT44. "
        "Mark as TS, run in REAL mode with LEO regime focus. Set priority at 15, conduct 4 hourly revisits. "
        "Begin at 2025-11-05 12:00:00+00:00 with 50-minute end offset. Configure for 3 frames per collection at 4 second exposure. "
        "Disable visibility checks and enable brightest UCT prioritization."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["22334", "55112", "78998"],
        "sensor_name_list": ["RME25", "LMNT44"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "LEO",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": 4.0,
        "number_of_frames": 3,
        "integration_time": 4,
        "end_time_offset_minutes": 50,
        "objective_start_time": "datetime.datetime(2025, 11, 5, 12, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
        "objective_uuid": None,
        "objective_end_time": None,
        "binning": None,
    },
    # 33. SingleIntentObjective
    (
        "Execute a SingleIntentObjective for target 45678 with RSO 98123 using detectors RME02 and LMNT11. "
        "Apply TS//FOUO classification in EXERCISE mode with SIDEREAL tracking at priority 6. "
        "Begin 2025-12-18 05:30:00+00:00 and terminate at 2025-12-18 07:45:00+00:00. "
        "Collect 7 frames with 1.5 second exposures using binning factor 4."
    ): {
        "classification_marking": "TS//FOUO",
        "target_id": "45678",
        "rso_id": "98123",
        "sensor_name_list": ["RME02", "LMNT11"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 7,
        "integration_time": 1.5,
        "priority": 6,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 12, 18, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 18, 7, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
    },
    # 34. DataEnrichmentObjective
    (
        "Commission a DataEnrichmentObjective for targets 34512, 78945, and 12358 employing sensors LMNT38 and RME41 "
        "with C marking in TEST environment. Configure for maximum 10 RSOs with 8 revisits hourly. "
        "Commence at 2026-02-10 17:45:00+00:00 and conclude by 2026-02-11 05:30:00+00:00. "
        "Assign priority value 15 and enforce visibility prerequisites."
    ): {
        "classification_marking": "C",
        "data_mode": "TEST",
        "target_id_list": ["34512", "78945", "12358"],
        "sensor_name_list": ["LMNT38", "RME41"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 2, 10, 17, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 2, 11, 5, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
        "objective_uuid": None,
        "binning": None,
        "intent_start_time": None,
        "intent_end_time": None,
    },
    # 35. SensorCheckoutObjective
    (
        "Authorize a SensorCheckoutObjective for instrument LMNT52 inspecting MEO region with S classification in SIMULATED mode. "
        "Implement SIDEREAL tracking at priority 8 with 2.5 hourly revisits. Schedule for 2026-03-25 13:10:00+00:00 "
        "allowing 40 minutes patience interval. Capture 4 frames with 5 second integration periods. Disable visibility requirements."
    ): {
        "classification_marking": "S",
        "sensor_name": "LMNT52",
        "orbital_regime": "MEO",
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 40,
        "revisits_per_hour": 2.5,
        "number_of_frames": 4,
        "integration_time": 5,
        "objective_start_time": "datetime.datetime(2026, 3, 25, 13, 10, tzinfo=TzInfo(UTC))",
        "priority": 8,
        "objective_name": "SensorCheckoutObjective",
        "objective_uuid": None,
        "objective_end_time": None,
        "binning": None,
        "intent_start_time": None,
        "intent_end_time": None,
    },
    # 36. BaselineAutonomyObjective
    (
        "Activate a BaselineAutonomyObjective with identifier '5f8a3d21-c9e7-4b1a-8a9c-f79d12345678'. "
        "Apply S//FOUO classification in REAL environment with priority 800. Track catalog entries 34567 and 89012 "
        "alongside RSO identifiers 44321, 55432, and 66543. Terminate operations at 2026-04-30 23:59:59+00:00."
    ): {
        "objective_uuid": "5f8a3d21-c9e7-4b1a-8a9c-f79d12345678",
        "classification_marking": "S//FOUO",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 800,
        "baseline_autonomy_rso": "34567,89012",
        "objective_end_time": "datetime.datetime(2026, 4, 30, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["44321", "55432", "66543"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 37: CatalogMaintenanceObjective
    (
        "I need a CatalogMaintenanceObjective scheduled for sensors RME08 and LMNT09 with TS classification marking. "
        "Configure it for SIMULATED mode operation with a high priority of 5. Make sure it begins at 2025-06-01 03:30:00+00:00 "
        "and concludes at 2025-06-01 08:45:00+00:00. We require a patience value of 15 minutes and an end time offset of 30 minutes. "
        "This should track objects in the MEO regime with RATE_TRACK_SIDEREAL tracking method. "
        "The RSO ID list should include '33221,44887,55129'. Please enable visibility checking."
    ): {
        "classification_marking": "TS",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 5,
        "sensor_name_list": ["RME08", "LMNT09"],
        "rso_id_list": ["33221", "44887", "55129"],
        "objective_start_time": "datetime.datetime(2025, 6, 1, 3, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 1, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 38: SearchObjective
    (
        "Please establish a SearchObjective for tracking target 98765 with the UKR03 sensor. Employ C classification marking and "
        "EXERCISE mode for this operation. Set up a CROSS_TRACK search type with high priority 3. The operation should commence at "
        "2025-08-15 14:45:00+00:00 and terminate at 2025-08-15 18:30:00+00:00. Configure initial offset to 45 seconds and final offset "
        "to 75 seconds with 60% frame overlap. Establish end time offset of 35 minutes and use 7 frames with 3.5 seconds integration time. "
        "Schedule search to begin 20 minutes after objective start. Visibility check must be enabled."
    ): {
        "classification_marking": "C",
        "target_id": "98765",
        "sensor_name": "UKR03",
        "search_type": "CROSS_TRACK",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 35,
        "priority": 3,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 8, 15, 14, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 15, 18, 30, tzinfo=TzInfo(UTC))",
        "number_of_frames": 7,
        "integration_time": 3.5,
        "search_start_time": "datetime.datetime(2025, 8, 15, 15, 5, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 39: GeodssRevisitObjective
    (
        "Setup a GeodssRevisitObjective for conducting observations of targets 22446 and 77993 utilizing sensors RME25 and LMNT28. "
        "Apply S classification and EXERCISE data mode. Implement SIDEREAL tracking at priority level 8 with patience window of 45 minutes. "
        "Objective should activate at 2025-07-06 23:15:00+00:00. Configure with readout_rate 0 (1MHz), gain_setting 1 (Low Gain), "
        "soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), "
        "binning_mode 0 (No Binning), and scan_mode 0 (Continuous). Set num_observations to 3."
    ): {
        "classification_marking": "S",
        "target_id_list": ["22446", "77993"],
        "sensor_name_list": ["RME25", "LMNT28"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 45,
        "revisits_per_hour": None,
        "number_of_frames": None,
        "integration_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 7, 6, 23, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 8,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "acquisition_type": 0,
        "auto_track_type": 2,
        "auto_track_roi_position": 0,
        "camera_mode": 1,
        "observation_interval": 0,
        "num_observations": 3,
        "num_skip_frames": 0,
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "rate_track_verify": 0,
        "soi_filter_position": 2,
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "overscan": 0,
        "command": 0,
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 40: PeriodicRevisitObjective
    (
        "Design a PeriodicRevisitObjective to track targets 55666, 77888, and 99000 with RME11 and LMNT14 sensors. "
        "Apply U//FOUO classification for this EXERCISE mode operation. We require 4 revisits per hour with 8 frames per visit "
        "and 1.5 seconds integration time. Schedule to commence at 2025-11-10 05:25:00+00:00 and end at 2025-11-10 17:45:00+00:00. "
        "Set binning to 4, priority to 7, and patience time to 25 minutes. Activate visibility checking and set optimal frames per hour to 350. "
        "This mission must ignore other objective intent submissions."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["55666", "77888", "99000"],
        "sensor_name_list": ["RME11", "LMNT14"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 25,
        "revisits_per_hour": 4.0,
        "number_of_frames": 8,
        "integration_time": 1.5,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 11, 10, 5, 25, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 10, 17, 45, tzinfo=TzInfo(UTC))",
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 41: UctObservationObjective
    (
        "Configure a UctObservationObjective for tracking uncorrelated targets 11122, 33344, and 55566 using sensors RME27 and LMNT29. "
        "Apply TS classification in REAL mode and focus on GEO regime objects. Set priority to 6 with 8 revisits per hour. "
        "Begin observations at 2025-12-05 22:10:00+00:00 and conclude at 2025-12-06 04:30:00+00:00. Configure with 3 frames per visit, "
        "2.5 seconds integration time, and binning level 2. Set end time offset to 40 minutes with patience time of 20 minutes. "
        "Disable sorting by brightest UCT but maintain visibility checking."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["11122", "33344", "55566"],
        "sensor_name_list": ["RME27", "LMNT29"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 2.5,
        "binning": 2,
        "end_time_offset_minutes": 40,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 22, 10, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 6, 4, 30, tzinfo=TzInfo(UTC))",
        "priority": 6,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 42: SingleIntentObjective
    (
        "Please set up a SingleIntentObjective for target 44556 and RSO 88999 using the LMNT33 sensor. Use C classification and "
        "apply SIMULATED mode with SIDEREAL tracking. The system should capture 10 frames with 4 seconds integration time and "
        "binning level 3. Assign priority level 4 for this task. Begin at 2025-09-18 10:40:00+00:00 and end at 2025-09-18 12:15:00+00:00. "
        "Also specify intent time window from 2025-09-18 11:00:00+00:00 to 2025-09-18 12:00:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id": "44556",
        "rso_id": "88999",
        "sensor_name_list": ["LMNT33"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 10,
        "integration_time": 4,
        "priority": 4,
        "binning": 3,
        "intent_start_time": "datetime.datetime(2025, 9, 18, 11, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 9, 18, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 18, 10, 40, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 18, 12, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 43: DataEnrichmentObjective
    (
        "Initiate a DataEnrichmentObjective for observing targets 12321, 45654, 78987 with RME40 and LMNT42 sensors. "
        "Set U marking and TEST mode while using RATE_TRACK tracking method. Configure to monitor a maximum of 10 RSOs with "
        "15 revisits per hour at priority level 15. Begin the objective at 2025-10-12 17:30:00+00:00. Include intent time window "
        "from 2025-10-12 18:00:00+00:00 to 2025-10-12 23:00:00+00:00. Set binning to 1 and disable visibility checking for this mission."
    ): {
        "classification_marking": "U",
        "data_mode": "TEST",
        "objective_uuid": None,
        "target_id_list": ["12321", "45654", "78987"],
        "sensor_name_list": ["RME40", "LMNT42"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "binning": 1,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 10, 12, 17, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "intent_start_time": "datetime.datetime(2025, 10, 12, 18, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 10, 12, 23, 0, tzinfo=TzInfo(UTC))",
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 44: SensorCheckoutObjective
    (
        "Establish a SensorCheckoutObjective for RME37 sensor focusing on the LEO orbital regime. Apply S//FOUO classification and "
        "utilize EXERCISE mode with RATE_TRACK collection method. Set 2.5 revisits per hour with 6 frames per visit and 1.8 seconds "
        "integration time. Begin checkout at 2025-05-08 12:20:00+00:00 and complete by 2025-05-08 18:45:00+00:00. Configure with "
        "binning level 2, patience time of 40 minutes, and priority 9. Disable visibility checking for this test."
    ): {
        "classification_marking": "S//FOUO",
        "sensor_name": "RME37",
        "orbital_regime": "LEO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 40,
        "revisits_per_hour": 2.5,
        "number_of_frames": 6,
        "integration_time": 1.8,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 5, 8, 12, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 8, 18, 45, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 9,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 45: BaselineAutonomyObjective
    (
        "Create a BaselineAutonomyObjective with UUID '7f8d9e0a-bc1d-4567-89ef-012345678901' using C//FOUO classification. "
        "Configure for EXERCISE mode with priority 800 and DARK frame type. Include catalog IDs 29876, 54321, and 13579 in the "
        "baseline autonomy RSO list. Add RSO IDs 44444, 55555, and 66666 to the tracking list. "
        "Set objective to conclude operations at 2025-08-30 23:59:59+00:00."
    ): {
        "objective_uuid": "7f8d9e0a-bc1d-4567-89ef-012345678901",
        "classification_marking": "C//FOUO",
        "data_mode": "EXERCISE",
        "frame_type": "DARK",
        "priority": 800,
        "baseline_autonomy_rso": "29876,54321,13579",
        "objective_end_time": "datetime.datetime(2025, 8, 30, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["44444", "55555", "66666"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 46: CatalogMaintenanceObjective
    (
        "Please define a CatalogMaintenanceObjective. This objective is intended for sensor RME08 and sensor LMNT09. "
        "It should carry a classification marking of C. The operational mode is SIMULATED. "
        "We need a priority level of 50. The patience duration before timeout is set to 15 minutes, and the end time offset should be 30 minutes. "
        "Crucially, the visibility check must be enabled. "
        "The objective will commence on 2025-06-10 at 10:00:00 UTC and conclude on 2025-06-10 at 14:00:00 UTC. "
        "The tracking methodology is RATE_TRACK within the MEO orbital regime. "
        "Furthermore, include RSO IDs '22334' and '88776'."
    ): {
        "classification_marking": "C",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["RME08", "LMNT09"],
        "rso_id_list": ["22334", "88776"],
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 47: SearchObjective
    (
        "Formulate a SearchObjective to locate target ID 'targ_alpha_001'. This search will be conducted by sensor 'XYZ77'. "
        "The classification marking must be 'TS'. The data mode will be 'EXERCISE'. Assign a high priority of 2. "
        "The tracking should use the 'SIDEREAL' method. The objective's window starts at 2025-08-01 12:00:00+00:00 and ends at 2025-08-01 15:30:00+00:00. "
        "Define an initial search offset of 45 seconds and a final offset of 75 seconds. The frame overlap needs to be precisely 0.6 (60%). "
        "The end time offset for scheduling is 50 minutes. "
        "The search type is 'CROSS_TRACK', and the search itself should begin 20 minutes after the objective's start time. "
        "Also, specify 10 frames with an integration time of 0.5 seconds and disable visibility check."
    ): {
        "classification_marking": "TS",
        "target_id": "targ_alpha_001",
        "sensor_name": "XYZ77",
        "search_type": "CROSS_TRACK",
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 50,
        "priority": 2,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 8, 1, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 1, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 0.5,
        "search_start_time": "datetime.datetime(2025, 8, 1, 12, 20, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 20 minutes
        "objective_name": "SearchObjective",
    },
    # New Example 48: GeodssRevisitObjective
    (
        "Construct a GeodssRevisitObjective targeting objects 'geo_target_01' and 'geo_target_02', utilizing sensor systems 'GEO_SNS_A' and 'GEO_SNS_B'. "
        "Apply a 'U//FOUO' classification marking. The system will operate in 'REAL' mode. "
        "This objective has a priority of 7. The specific tracking approach is 'RATE_TRACK'. "
        "Set the objective to start on 2025-09-15 08:00:00+00:00 and end on 2025-09-15 12:00:00+00:00. "
        "Configure the GEODSS parameters as follows: readout_rate 0 (1MHz), gain_setting 1 (Low Gain), "
        "soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), "
        "array_kind 1 (Photometer), binning_mode 0 (No Binning), and scan_mode 0 (Continuous). "
        "Set number of frames to 10 and integration time to 1.5 seconds. Set visibility check to true."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["geo_target_01", "geo_target_02"],
        "sensor_name_list": ["GEO_SNS_A", "GEO_SNS_B"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 1.5,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 15, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 15, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": False,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "acquisition_type": 0,  # Default
        "auto_track_type": 2,
        "auto_track_roi_position": 0,  # Default
        "camera_mode": 1,
        "observation_interval": 0.0,  # Default
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "rate_track_verify": 0,  # Default
        "soi_filter_position": 2,
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "overscan": 0,  # Default
        "command": 0,  # Default
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 49: PeriodicRevisitObjective
    (
        "Develop a PeriodicRevisitObjective for target IDs 'periodic_A' and 'periodic_B', employing sensors 'SNS_P1' and 'SNS_P2'. "
        "This objective is marked 'S', operates in 'TEST' mode, and has a priority of 3. "
        "Patience before failure is set at 20 minutes. We desire 4 revisits per hour for these targets. "
        "The objective is to commence at 2025-07-01 00:00:00 UTC. "
        "The system should capture 3 frames per intent, each with an integration time of 2.5 seconds. Set binning to 2. "
        "Ensure that intent submissions from other objectives are not ignored. The visibility check for RSOs should be true. "
        "The objective will end on 2025-07-01 06:00:00 UTC. "
        "The intent start time is 2025-07-01 01:00:00+00:00 and intent end time is 2025-07-01 05:00:00+00:00."
    ): {
        "classification_marking": "S",
        "target_id_list": ["periodic_A", "periodic_B"],
        "sensor_name_list": ["SNS_P1", "SNS_P2"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 4.0,
        "number_of_frames": 3,
        "integration_time": 2.5,
        "binning": 2,
        "objective_start_time": "datetime.datetime(2025, 7, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 7, 1, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 3,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,
        "intent_start_time": "datetime.datetime(2025, 7, 1, 1, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 7, 1, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 50: UctObservationObjective
    (
        "Generate a UctObservationObjective specifically for UCT RSO IDs 'uct_obj_X1' and 'uct_obj_Y2'. "
        "The sensors assigned are 'UCT_SENS_1' and 'UCT_SENS_2'. "
        "This carries a 'U//FOUO' marking and operates in 'REAL' data mode, within the LEO orbital regime. "
        "Assign a priority of 9. The objective aims for 8.0 revisits per hour. "
        "The start time is 2025-10-20 05:30:00+00:00. Crucially, enable sorting by the brightest UCT. "
        "The end time offset is 45 minutes, and the visibility check is disabled. "
        "Configure 4 frames per intent with an integration time of 1.8 seconds. Binning should be 1."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["uct_obj_X1", "uct_obj_Y2"],
        "sensor_name_list": ["UCT_SENS_1", "UCT_SENS_2"],
        "data_mode": "REAL",  # Default
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "LEO",
        "visibility_check": False,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 8.0,
        "number_of_frames": 4,
        "integration_time": 1.8,
        "binning": 1,
        "end_time_offset_minutes": 45,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 20, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 9,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 51: SingleIntentObjective
    (
        "Fashion a SingleIntentObjective. This objective should reference target ID 'single_TGT_007' and RSO ID 'single_RSO_007'. "
        "It will utilize sensors 'SENSOR_ALPHA' and 'SENSOR_BETA'. "
        "The classification marking is 'U', and the data mode is 'SIMULATED'. The tracking type specified is 'RATE_TRACK'. "
        "Assign a priority of 6. The objective is set to start on 2025-11-05 18:00:00 UTC and end on 2025-11-05 20:00:00 UTC. "
        "For this intent, capture 7 frames, each with an integration time of 3 seconds. Set camera binning to 4. "
        "The intent should start no earlier than 2025-11-05 18:15:00+00:00 and end no later than 2025-11-05 19:15:00+00:00."
    ): {
        "classification_marking": "U",
        "target_id": "single_TGT_007",
        "rso_id": "single_RSO_007",
        "sensor_name_list": ["SENSOR_ALPHA", "SENSOR_BETA"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 7,
        "integration_time": 3.0,
        "priority": 6,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2025, 11, 5, 18, 15, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 11, 5, 19, 15, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 5, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 5, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 52: DataEnrichmentObjective
    (
        "Create a DataEnrichmentObjective. This objective is for enriching data on target IDs 'enrich_T1', 'enrich_T2', and 'enrich_T3'. "
        "Sensors 'ENRICH_S_A' and 'ENRICH_S_B' are to be used. The classification marking is 'C'. "
        "The data mode is 'EXERCISE'. Employ 'SIDEREAL' tracking. The objective will attempt to observe a maximum of 10 RSOs. "
        "Aim for 15 revisits per hour. The objective is scheduled to begin on 2026-01-10 03:00:00+00:00. "
        "The visibility check should be disabled. The objective will end on 2026-01-10 09:00:00+00:00. Priority is 25."
    ): {
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["enrich_T1", "enrich_T2", "enrich_T3"],
        "sensor_name_list": ["ENRICH_S_A", "ENRICH_S_B"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 3, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 10, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 53: SensorCheckoutObjective
    (
        "Initiate a SensorCheckoutObjective with a classification marking of 'S'. This checkout is for sensor 'CHK_SNS_04'. "
        "The operational mode is 'TEST', and the target orbital regime is 'XGEO'. Tracking should be 'RATE_TRACK_SIDEREAL'. "
        "Assign this a priority of 4. The system should aim for 0.5 revisits per hour. "
        "The objective starts on 2026-02-15 12:30:00+00:00 and will conclude by 2026-02-16 00:30:00+00:00. "
        "The visibility check must be true. Patience is 45 minutes. "
        "Capture 2 frames per observation with an integration time of 5 seconds. Set binning to null."
    ): {
        "classification_marking": "S",
        "sensor_name": "CHK_SNS_04",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 45,
        "revisits_per_hour": 0.5,
        "number_of_frames": 2,
        "integration_time": 5.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 15, 12, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 2, 16, 0, 30, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 4,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 54: BaselineAutonomyObjective
    (
        "Define a BaselineAutonomyObjective. This objective requires a specific UUID: 'abcdef01-2345-6789-abcd-ef0123456789'. "
        "It will operate with 'TS' classification markings and in 'SIMULATED' data mode. "
        "The frame type is 'DARK'. Set the priority to 1500. "
        "The list of baseline autonomy catalog IDs to track is 'cat_id_001,cat_id_002,cat_id_003'. "
        "This objective should have no specific end time, allowing it to run continuously. "
        "Also provide a list of RSO IDs: 'rso_base_A', 'rso_base_B'."
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "TS",
        "data_mode": "SIMULATED",
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "cat_id_001,cat_id_002,cat_id_003",
        "objective_end_time": None,
        "rso_id_list": ["rso_base_A", "rso_base_B"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 55
    (
        "Initiate a CatalogMaintenanceObjective using sensors 'SENSRX1' and 'SENSRY2'. The operation requires 'S' classification markings and should run in 'REAL' data mode. We need to set a priority level of 75, a patience duration of 15 minutes, and an end time offset of 30 minutes. Please ensure visibility check is activated. The objective is scheduled to commence on 2025-11-10 at 08:00:00+00:00 and conclude by 2025-11-10 at 12:30:00+00:00. Employ 'RATE_TRACK' tracking, focusing on the 'MEO' orbital regime. The RSO ID list for this task includes '33001', '33002', and '33003'. Specify binning as 2x2."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 75,
        "sensor_name_list": ["SENSRX1", "SENSRY2"],
        "rso_id_list": ["33001", "33002", "33003"],
        "objective_start_time": "datetime.datetime(2025, 11, 10, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 10, 12, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": 2,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 56
    (
        "We urgently need to create a SearchObjective for target 'TGT-ALPHA-007' using the 'RAZOR-EYE-03' sensor. This is a 'TS' (Top Secret) level operation, to be conducted in 'SIMULATED' mode with the highest priority of 1. The tracking should be 'SIDEREAL'. Define the objective to start on 2026-01-15 at 03:00:00+00:00 and end precisely on 2026-01-15 at 04:00:00+00:00. The initial search offset needs to be 120 seconds, with a final offset of 150 seconds. Ensure a frame overlap of exactly 25% (0.25). The end time offset for scheduling purposes should be 10 minutes. Utilize a 'CROSS_TRACK' search pattern. The actual search should commence 10 minutes after the objective's defined start time. Also, set 10 frames with an integration time of 0.5 seconds."
    ): {
        "classification_marking": "TS",
        "target_id": "TGT-ALPHA-007",
        "sensor_name": "RAZOR-EYE-03",
        "search_type": "CROSS_TRACK",
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 120,
        "final_offset": 150,
        "frame_overlap_percentage": 0.25,
        "end_time_offset_minutes": 10,
        "priority": 1,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 3, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 15, 4, 0, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 0.5,
        "search_start_time": "datetime.datetime(2026, 1, 15, 3, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 57
    (
        "Configure a GeodssRevisitObjective for the targets 'CRIT-TGT-X1', 'CRIT-TGT-Y2', and 'CRIT-TGT-Z3'. This task will utilize the 'GEODSS-SITE-A' and 'GEODSS-SITE-B' sensors. The classification marking is 'U//FOUO'. The objective is set to operate in 'REAL' data mode with a priority of 5. The tracking method will be 'RATE_TRACK'. The objective should start on 2025-12-01 at 22:00:00+00:00 and will have no explicit end time, meaning it runs indefinitely or until manually stopped. For GEODSS specific settings: set acquisition type to 2 (Auto Rate Track), auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), observation interval to 5.5 seconds, and number of observations to 3. Furthermore, the readout rate should be 0 (1MHz), gain setting 1 (Low Gain), SOI filter position 2 (10% Light), array kind 1 (Photometer), HW binning mode (1), and continuous scan mode (0). We also need to specify 10 revisits per hour, 4 frames per intent, and an integration time of 1.2 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["CRIT-TGT-X1", "CRIT-TGT-Y2", "CRIT-TGT-Z3"],
        "sensor_name_list": ["GEODSS-SITE-A", "GEODSS-SITE-B"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": 10.0,
        "number_of_frames": 4,
        "integration_time": 1.2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 22, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "acquisition_type": 2,
        "auto_track_type": 2,
        "auto_track_roi_position": 0,
        "camera_mode": 1,
        "observation_interval": 5.5,
        "num_observations": 3,
        "num_skip_frames": 0,
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "rate_track_verify": 0,
        "soi_filter_position": 2,
        "array_kind": 1,
        "binning_mode": 1,
        "scan_mode": 0,
        "overscan": 0,
        "command": 0,
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 58
    (
        "Please set up a PeriodicRevisitObjective. This objective is for targets with IDs 'PR-TGT-001' and 'PR-TGT-002', using sensors 'EYE-IN-SKY-A' and 'EYE-IN-SKY-B'. The classification marking for this operation should be 'C' (Confidential), and it must operate in 'EXERCISE' mode. Assign it a priority of 15. The patience for intent failures is 45 minutes. Crucially, this objective should *not* ignore intent submissions from other objectives. Schedule this to begin on 2026-03-10 at 00:00:00+00:00 and to end on 2026-03-24 at 00:00:00+00:00. We are aiming for 5.5 revisits per hour, with each intent capturing 8 frames, and an integration time of 2.5 seconds per frame. Let's also set binning to 4 and optimal frames per hour to 350. Visibility check should be true."
    ): {
        "classification_marking": "C",
        "target_id_list": ["PR-TGT-001", "PR-TGT-002"],
        "sensor_name_list": ["EYE-IN-SKY-A", "EYE-IN-SKY-B"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 5.5,
        "number_of_frames": 8,
        "integration_time": 2.5,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 3, 24, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 59
    (
        "We need to create a UctObservationObjective. This is for observing Uncorrelated Tracks (UCTs) with RSO IDs 'UCT-X990', 'UCT-Y880', and 'UCT-Z770'. Please assign sensors 'DEEPSTAR-1' and 'DEEPSTAR-2' for this task. The data should be marked 'U//FOUO' and processed in 'REAL' mode. The operational orbital regime is 'XGEO'. Set a high priority of 7 for this objective. We require exactly 4.0 revisits per hour. The objective is to commence on 2026-02-20 at 10:30:00+00:00 and will not have a specified end time. Critically, enable sorting by the brightest UCT. The end time offset for scheduling individual intents should be 45 minutes. Ensure visibility check is disabled for this. Each observation should capture 3 frames with an integration time of 3 seconds. No binning is required."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["UCT-X990", "UCT-Y880", "UCT-Z770"],
        "sensor_name_list": ["DEEPSTAR-1", "DEEPSTAR-2"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": 4.0,
        "number_of_frames": 3,
        "integration_time": 3.0,
        "binning": None,
        "end_time_offset_minutes": 45,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 10, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 60
    (
        "Construct a SingleIntentObjective. This objective needs to target RSO ID 'RSO-DELTA-42'. The observation will be carried out using sensor 'PINPOINT-07'. The classification marking is 'S', and the data mode should be 'TEST'. Use 'RATE_TRACK' for the collection request type. Assign this a priority level of 3. The objective should start on 2025-10-05 at 15:00:00+00:00 and conclude on 2025-10-05 at 15:15:00+00:00. For this specific intent, capture exactly 1 frame with an integration time of 10.5 seconds. Set camera binning to 1x1 (value 1)."
    ): {
        "classification_marking": "S",
        "target_id": None,
        "rso_id": "RSO-DELTA-42",
        "sensor_name_list": ["PINPOINT-07"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 1,
        "integration_time": 10.5,
        "priority": 3,
        "binning": 1,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 5, 15, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 5, 15, 15, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 61
    (
        "Establish a DataEnrichmentObjective. The objective will focus on enriching data for target IDs 'ENRICH-01', 'ENRICH-02', 'ENRICH-03', and 'ENRICH-04'. We will employ sensors 'DATASCOPE-A' and 'DATASCOPE-B' for this purpose. The classification level must be 'U', and the data mode is 'SIMULATED'. The collection request type should be 'SIDEREAL'. Configure the objective to observe a maximum of 5 RSOs and aim for 15.0 revisits per hour. The objective is scheduled to start on 2026-04-01 at 06:00:00+00:00. The visibility check for RSOs must be enabled. Let's set the priority to 25. This objective has no defined end time."
    ): {
        "classification_marking": "U",
        "data_mode": "SIMULATED",
        "objective_uuid": None,
        "target_id_list": ["ENRICH-01", "ENRICH-02", "ENRICH-03", "ENRICH-04"],
        "sensor_name_list": ["DATASCOPE-A", "DATASCOPE-B"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 4, 1, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 62
    (
        "Please define a SensorCheckoutObjective with a 'U//FOUO' classification marking for the 'NEW-SAT-IMAGER-01' sensor. The data mode will be 'TEST'. We are targeting the 'LEO' orbital regime for this checkout. Use 'RATE_TRACK' as the collect request type. Set the priority to 5. The checkout objective demands 0.5 revisits per hour and should commence on 2025-09-15 at 12:00:00+00:00, with an objective end time of 2025-09-16 at 12:00:00+00:00. Visibility check must be set to false. The patience for this checkout is 20 minutes. It should capture 10 frames per intent, with an integration time of 0.25 seconds. Set binning to 2."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "NEW-SAT-IMAGER-01",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 20,
        "revisits_per_hour": 0.5,
        "number_of_frames": 10,
        "integration_time": 0.25,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 15, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 16, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 63
    (
        "Create a BaselineAutonomyObjective, and you must assign it the specific UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. This objective will operate with 'C' classification markings. The data mode needs to be 'EXERCISE', and the frame type should be 'DARK'. Set a custom priority of 1500. For the baseline autonomy, it needs to track catalog IDs '25544' (ISS), '27601', and '43400'. This objective is intended for continuous operation, so no objective end time should be specified. Include an initial RSO ID list with 'SATCAT-A', 'SATCAT-B', even though it might be updated later."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "25544,27601,43400",
        "objective_end_time": None,
        "rso_id_list": ["SATCAT-A", "SATCAT-B"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 64: CatalogMaintenanceObjective
    (
        "I need a new CatalogMaintenanceObjective configured for sensors KVN08 and DSBR09. Please use TS classification marking and REAL data mode. Set the priority to 8 with patience of 15 minutes. Schedule it to start tomorrow at 2025-05-15 14:30:00+00:00 and run until 2025-05-15 20:45:00+00:00. Apply MEO orbital regime with SIDEREAL tracking method. The visibility check should be enabled. Include RSO IDs '98765' and '43210' in the tracking list."
    ): {
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 20,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 5, 15, 20, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 8,
        "rso_id_list": ["98765", "43210"],
        "sensor_name_list": ["KVN08", "DSBR09"],
        "visibility_check": True,
        "binning": None,
    },
    # New Example 65: SearchObjective
    (
        "Set up a SearchObjective for target 45678 using the JPN03 sensor. Configuration should include C marking and SIMULATED mode with priority 3. Use RATE_TRACK tracking type with CROSS_TRACK search pattern. Begin operations on 2025-05-16 06:15:00+00:00 and terminate at 2025-05-16 08:45:00+00:00. Set initial offset to 45 seconds and final offset to 75 seconds with frame overlap of 65%. Enable visibility checking and assign a search start time of 2025-05-16 06:45:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id": "45678",
        "sensor_name": "JPN03",
        "search_type": "CROSS_TRACK",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.65,
        "end_time_offset_minutes": 20,
        "priority": 3,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 5, 16, 6, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 16, 8, 45, tzinfo=TzInfo(UTC))",
        "search_start_time": "datetime.datetime(2025, 5, 16, 6, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 66: GeodssRevisitObjective
    (
        "Please establish a GeodssRevisitObjective for monitoring objects 34567 and 89012. Employ sensors GAV25 and RTL19 with S//FOUO classification level. Keep default RATE_TRACK_SIDEREAL tracking. Begin surveillance at 2025-05-17 22:10:00+00:00. Configure camera settings with readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), scan_mode 0 (Continuous)."
    ): {
        "classification_marking": "S//FOUO",
        "target_id_list": ["34567", "89012"],
        "sensor_name_list": ["GAV25", "RTL19"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "objective_start_time": "datetime.datetime(2025, 5, 17, 22, 10, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 10,
        "acquisition_type": 0,
        "auto_track_type": 2,
        "auto_track_roi_position": 0,
        "camera_mode": 1,
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "soi_filter_position": 2,
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "objective_name": "GeodssRevisitObjective",
        "visibility_check": False,
        "patience_minutes": 30,
    },
    # New Example 67: PeriodicRevisitObjective
    (
        "Deploy a PeriodicRevisitObjective to continuously monitor satellites 56789 and 23456. Utilize the PAC10 and CHI15 sensors with TS//FOUO security classification. Switch to EXERCISE data mode. Begin operation on 2025-05-20 03:45:00+00:00 and continue indefinitely. Configure for 5 revisits per hour, 3 frames per visit with 1.5 seconds integration time. Employ RATE_TRACK collection method and set priority level to 5. Enable visibility checks and ensure optimal frames per hour is set to 350."
    ): {
        "classification_marking": "TS//FOUO",
        "target_id_list": ["56789", "23456"],
        "sensor_name_list": ["PAC10", "CHI15"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 5.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "objective_start_time": "datetime.datetime(2025, 5, 20, 3, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 68: UctObservationObjective
    (
        "Initiate UCT observation for unidentified objects 78901, 23456, and 34567 in the XGEO region. Assign sensors ALT11 and SYD22 with U marking and TEST mode. Prioritize at level 7 and schedule 8 revisits hourly, starting June 1st at 2025-06-01 15:20:00+00:00. Configure with 4 frames per observation using 3 second exposures. Enable sorting by brightness to focus on most visible objects first. Set end time offset to 90 minutes and disable visibility checking."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["78901", "23456", "34567"],
        "sensor_name_list": ["ALT11", "SYD22"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": 8.0,
        "number_of_frames": 4,
        "integration_time": 3.0,
        "end_time_offset_minutes": 90,
        "objective_start_time": "datetime.datetime(2025, 6, 1, 15, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 69: SingleIntentObjective
    (
        "Execute a SingleIntentObjective for immediate observation of target 77889 with RSO 66554. Deploy using the HWI49 and TPE20 sensors at C//FOUO classification. Schedule for next Monday at 2025-05-19 05:30:00+00:00 and conclude by 2025-05-19 07:45:00+00:00. Configure for 7 frames with 4.5 second exposures and binning level 4. Set to SIMULATED mode with SIDEREAL tracking at priority 4."
    ): {
        "classification_marking": "C//FOUO",
        "target_id": "77889",
        "rso_id": "66554",
        "sensor_name_list": ["HWI49", "TPE20"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 7,
        "integration_time": 4.5,
        "priority": 4,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 5, 19, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 19, 7, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 70: DataEnrichmentObjective
    (
        "Implement a DataEnrichmentObjective for detailed analysis of targets 12340, 56780, and 90120. The observatory will use KSC30 and JDK17 sensors with S classification for secure processing. Commence at 2025-05-25 11:00:00+00:00 with no predetermined end time. Set parameters to observe maximum 10 RSOs with 15 revisits per hour at priority 15. Use RATE_TRACK collection method in REAL operational mode. Disable visibility checks for continuous coverage."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "target_id_list": ["12340", "56780", "90120"],
        "sensor_name_list": ["KSC30", "JDK17"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 5, 25, 11, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 71: SensorCheckoutObjective
    (
        "Schedule a SensorCheckoutObjective for the newly installed LRD55 sensor. Apply U//FOUO classification with EXERCISE mode for testing. Focus on LEO regime observations using RATE_TRACK method at priority 6. Begin checkout process at 2025-05-30 18:00:00+00:00 and run until 2025-05-31 06:00:00+00:00. Configure for 2 revisits per hour with 6 frames per observation and 2.5 second integration time. Apply binning of 2 for improved signal-to-noise ratio during checkout."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "LRD55",
        "orbital_regime": "LEO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 2.0,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "binning": 2,
        "objective_start_time": "datetime.datetime(2025, 5, 30, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 31, 6, 0, tzinfo=TzInfo(UTC))",
        "priority": 6,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 72: BaselineAutonomyObjective
    (
        "Activate a BaselineAutonomyObjective with UUID '789f9876-d45c-78a1-b234-123456789abc' for autonomous operations. Set U//FOUO classification in REAL mode with highest surveillance priority of 1200. Monitor catalog IDs 98765, 43210, and 12345 continuously with no end date. Add associated RSO IDs 55555, 66666, and 77777 to tracking list for comprehensive coverage."
    ): {
        "objective_uuid": "789f9876-d45c-78a1-b234-123456789abc",
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1200,
        "baseline_autonomy_rso": "98765,43210,12345",
        "objective_end_time": None,
        "rso_id_list": ["55555", "66666", "77777"],
        "objective_name": "BaselineAutonomyObjective",
    },
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

        # Aggregate every incorrect‐field instance across runs
        all_errors = []
        for r in self.results:
            for field_name, info in r["field_details"].items():
                if not info["correct"]:
                    all_errors.append(
                        {
                            # "query": r["query"],
                            # "objective_name": r["expected_objective_name"],
                            "field": field_name,
                            "expected": info["expected"],
                            "predicted": info["predicted"],
                        }
                    )

        report["all_errors"] = all_errors

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

    benchmark.print_results_analysis()

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
