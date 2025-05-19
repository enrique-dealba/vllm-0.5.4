import asyncio
import copy
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
    # New Example 73: CatalogMaintenanceObjective
    (
        "Initiate a CatalogMaintenanceObjective to monitor specific space objects. This objective will utilize sensors ALPH01 and BETA07. "
        "It's designated with C markings and will operate in SIMULATED mode. The priority for this task is set to 50. "
        "We require a patience duration of 15 minutes before considering an intent failed, and the end time offset should be 30 minutes. "
        "The objective needs to commence on 2025-11-01 at 10:00:00 UTC and is scheduled to conclude on 2025-11-01 at 14:30:00 UTC. "
        "The tracking method will be SIDEREAL, focusing on the MEO orbital regime. "
        "Please ensure that the visibility check is enabled. The RSO IDs for this objective are '25544' and '25545'."
    ): {
        "classification_marking": "C",
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["ALPH01", "BETA07"],
        "rso_id_list": ["25544", "25545"],
        "objective_start_time": "datetime.datetime(2025, 11, 1, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 1, 14, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 74: SearchObjective
    (
        "We need to create a SearchObjective for a specific target, identified by UUID 'target-uuid-001', employing sensor GAMMA03. "
        "This operation is classified with S markings and will proceed in REAL data mode. The assigned priority is critical, set at 2. "
        "The tracking should be RATE_TRACK. The objective will be active starting from 2025-12-05 08:00:00+00:00 and will run until 2025-12-05 10:00:00+00:00. "
        "Configure an initial search offset of 45 seconds and a final offset of 75 seconds from the RSO's current state. "
        "A frame overlap of 60% is required. The end time offset for scheduling the intent is 35 minutes. "
        "Utilize a CROSS_TRACK search pattern. The search itself should commence 10 minutes after the objective_start_time. "
        "Set the number of frames to 10 and integration time to 0.5 seconds. Visibility check should be disabled."
    ): {
        "classification_marking": "S",
        "target_id": "target-uuid-001",
        "sensor_name": "GAMMA03",
        "search_type": "CROSS_TRACK",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 35,
        "priority": 2,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 5, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 0.5,
        "search_start_time": "datetime.datetime(2025, 12, 5, 8, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 75: GeodssRevisitObjective
    (
        "Establish a GeodssRevisitObjective. This is for targets 'target-id-geo-007' and 'target-id-geo-008'. "
        "The sensors assigned are DELTA09 and EPSLN11. This objective carries a U//FOUO classification marking. "
        "It must operate in REAL mode. The priority level is 7. The tracking method is RATE_TRACK_SIDEREAL. "
        "Commence this objective on 2026-01-15 at 20:00:00 UTC. "
        "For camera settings: set readout_rate to 0 (1MHz), gain_setting to 1 (Low Gain), "
        "soi_filter to 2 (10% Light), auto_track_type to 2 (Manual), camera_mode to 1 (Zoomed EBS), "
        "array_kind to 0 (Main), binning_mode to 0 (No Binning), and scan_mode to 0 (Continuous). "
        "Set num_observations to 3 and observation_interval to 15.5 seconds. The patience is 25 minutes. "
        "Specify 3 revisits per hour. Set the objective end time to 2026-01-16 02:00:00 UTC."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["target-id-geo-007", "target-id-geo-008"],
        "sensor_name_list": ["DELTA09", "EPSLN11"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default
        "patience_minutes": 25,
        "revisits_per_hour": 3.0,
        "number_of_frames": None,  # Not specified, so None
        "integration_time": None,  # Not specified, so None
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 16, 2, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": False,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "acquisition_type": 0,  # Default
        "auto_track_type": 2,
        "auto_track_roi_position": 0,  # Default
        "camera_mode": 1,
        "observation_interval": 15.5,
        "num_observations": 3,
        "num_skip_frames": 0,  # Default
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "rate_track_verify": 0,  # Default
        "soi_filter_position": 2,
        "array_kind": 0,
        "binning_mode": 0,
        "scan_mode": 0,
        "overscan": 0,  # Default
        "command": 0,  # Default
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 76: PeriodicRevisitObjective
    (
        "Please configure a PeriodicRevisitObjective. The targets for this are 'periodic-target-A' and 'periodic-target-B'. "
        "Sensors ZETA99 and ETA08 are to be used. The classification is S. This will be a TEST mode operation. "
        "Set the priority to 15. Patience window is 40 minutes. Let's not ignore other objective intent submissions. "
        "The objective should begin on 2026-02-20 at 05:30:00+00:00. We need an optimal frames per hour rate of 300. "
        "Capture 3 frames per intent. Each frame requires an integration time of 1.5 seconds. Set binning to 2x2. "
        "This objective does not have a defined end time. Set revisits per hour to 4.5. Visibility check should be activated."
    ): {
        "classification_marking": "S",
        "target_id_list": ["periodic-target-A", "periodic-target-B"],
        "sensor_name_list": ["ZETA99", "ETA08"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 4.5,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": 2,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 77: UctObservationObjective
    (
        "Generate a UctObservationObjective. This is for UCT RSOs 'uct-alpha-001', 'uct-beta-002'. The sensors involved are THETA01 and IOTA02. "
        "Mark this with TS classification. The data mode is EXERCISE. This objective targets the GEO orbital regime. "
        "Priority is set to 5. We need 5.0 revisits per hour for these UCTs. "
        "The objective starts on 2026-03-10 12:00:00+00:00. Enable sorting by the brightest UCT. "
        "The end time offset for scheduling intents will be 75 minutes. Visibility check must be true. "
        "Set the number of frames per intent to 4, and an integration time of 2.5 seconds. Patience is 20 minutes."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["uct-alpha-001", "uct-beta-002"],
        "sensor_name_list": ["THETA01", "IOTA02"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 5.0,
        "number_of_frames": 4,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default
        "priority": 5,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 78: SingleIntentObjective
    (
        "Let's create a SingleIntentObjective. The target ID is 'single-target-X1', and the RSO ID is 'rso-Y2'. "
        "This will use sensors KAPPA11 and LAMDA12. The classification marking is U. Data mode should be SIMULATED. "
        "Employ RATE_TRACK tracking. The priority is relatively high, at 8. "
        "The objective is to start on 2026-04-05 at 15:45:00 UTC. We need 2 frames, each with 3.0 seconds of integration time. Set binning to 4. "
        "The intent should start no earlier than 2026-04-05 16:00:00 UTC and end no later than 2026-04-05 18:00:00 UTC. "
        "The overall objective should conclude by 2026-04-05 19:00:00 UTC."
    ): {
        "classification_marking": "U",
        "target_id": "single-target-X1",
        "rso_id": "rso-Y2",
        "sensor_name_list": ["KAPPA11", "LAMDA12"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 2,
        "integration_time": 3.0,
        "priority": 8,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2026, 4, 5, 16, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 4, 5, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 15, 45, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 4, 5, 19, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 79: DataEnrichmentObjective
    (
        "A DataEnrichmentObjective is required for several targets: 'enrich-T101', 'enrich-T102', and 'enrich-T103'. "
        "This objective will be handled by sensors MU20 and NU21. Set the classification marking to C. Use REAL data mode. "
        "The tracking type will be SIDEREAL. We aim to observe a maximum of 5 RSOs for enrichment. "
        "The desired rate is 8.0 revisits per hour. The objective should start on 2026-05-12 at 09:00:00+00:00. "
        "Ensure visibility check is true. The priority for this task is 25. No specific binning is needed. "
        "The objective's operational window will close on 2026-05-12 17:00:00+00:00."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["enrich-T101", "enrich-T102", "enrich-T103"],
        "sensor_name_list": ["MU20", "NU21"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 5, 12, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 5, 12, 17, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 80: SensorCheckoutObjective
    (
        "Please define a SensorCheckoutObjective. The classification for this is U//FOUO. The sensor to be checked out is XI30. "
        "This checkout will be in the LEO orbital regime. Data mode must be TEST. Use RATE_TRACK_SIDEREAL tracking. "
        "Priority is 12. We want 0.5 revisits per hour. The objective starts on 2026-06-01 00:00:00 UTC. "
        "Visibility check should be enabled. Patience is 35 minutes. Request 6 frames per intent with an integration time of 1.0 second. "
        "Binning is not required. The intent start time should be 2026-06-01 00:15:00 UTC and end by 2026-06-01 03:15:00 UTC."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "XI30",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 35,
        "revisits_per_hour": 0.5,
        "number_of_frames": 6,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 6, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default
        "intent_start_time": "datetime.datetime(2026, 6, 1, 0, 15, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 6, 1, 3, 15, 0, tzinfo=TzInfo(UTC))",
        "priority": 12,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 81: BaselineAutonomyObjective
    (
        "We are setting up a BaselineAutonomyObjective. This objective is identified by the UUID 'baseline-autonomy-guid-789'. "
        "It operates with U classification markings. The data mode is REAL. Frame type will be LIGHT. "
        "The priority for this continuous operation is set to 1500. "
        "The baseline autonomy RSO catalog IDs to track are '34567,89012,11223'. "
        "There is no specific end time for this objective, it should run continuously. "
        "Include initial RSO IDs 'rso-base-01' and 'rso-base-02', though these will be updated at runtime."
    ): {
        "objective_uuid": "baseline-autonomy-guid-789",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1500,
        "baseline_autonomy_rso": "34567,89012,11223",
        "objective_end_time": None,
        "rso_id_list": ["rso-base-01", "rso-base-02"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 82: CatalogMaintenanceObjective
    (
        "I need to establish a CatalogMaintenanceObjective with TS classification for MEO regime. Please use sensors AFB09 and STRC15, "
        "set to REAL mode with priority 8, patience window of 15 minutes, and end time offset of 35 minutes. Enable visibility checking. "
        "Schedule this to begin at 2025-03-15 14:30:00+00:00 and conclude at 2025-03-15 18:45:00+00:00. Use RATE_TRACK_SIDEREAL tracking. "
        "Include RSO IDs '33456, 78901, 45678' in the tracking list."
    ): {
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 3, 15, 18, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 3, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 8,
        "rso_id_list": ["33456", "78901", "45678"],
        "sensor_name_list": ["AFB09", "STRC15"],
        "visibility_check": True,
        "binning": None,
    },
    # New Example 83: SearchObjective
    (
        "We require a SearchObjective targeting ID 98765 using the LMNT33 sensor platform. Apply C classification marking and operate in TEST mode with priority 3. "
        "Configure initial offset to 45 seconds and final offset to 120 seconds with a 60% frame overlap. Set the end time offset to 30 minutes. "
        "Plan operations from 2025-05-30 22:15:00+00:00 to 2025-05-31 01:45:00+00:00 using SIDEREAL tracking method. "
        "The search should be CROSS_TRACK type with search start time 20 minutes after objective start. Include visibility checking functionality."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 30,
        "final_offset": 120,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 5, 31, 1, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 30, 22, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 5, 30, 22, 35, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "LMNT33",
        "target_id": "98765",
        "visibility_check": True,
    },
    # New Example 84: GeodssRevisitObjective
    (
        "Configure a GeodssRevisitObjective for objects 55555 and 77777 employing sensors AFB11 and GEOS02. Utilize C//FOUO marking in TEST environment. "
        "Begin at 2025-06-07 03:45:00+00:00. Set priority to 7 with RATE_TRACK_SIDEREAL approach. Configure camera with the following: "
        "readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 2 (Manual), "
        "camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), scan_mode 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "C//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "TEST",
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
        "objective_start_time": "datetime.datetime(2025, 6, 7, 3, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 30,
        "priority": 7,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["AFB11", "GEOS02"],
        "soi_filter_position": 2,
        "target_id_list": ["55555", "77777"],
        "visibility_check": False,
    },
    # New Example 85: PeriodicRevisitObjective
    (
        "Please set up a PeriodicRevisitObjective for tracking RSOs 44332 and 11009 using sensor arrays LMNT09 and RME12. "
        "Use TS//FOUO classification in EXERCISE mode. Schedule for 8 revisits per hour, starting 2025-04-17 06:00:00+00:00 until 2025-04-18 06:00:00+00:00. "
        "Set priority level 5, patience time 45 minutes, and enable visibility checking. Configure with 3 frames per visit and 1.5 seconds integration time. "
        "Employ SIDEREAL tracking method with ignore_other_objective_intent_submissions enabled."
    ): {
        "classification_marking": "TS//FOUO",
        "target_id_list": ["44332", "11009"],
        "sensor_name_list": ["LMNT09", "RME12"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 4, 17, 6, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 4, 18, 6, 0, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 86: UctObservationObjective
    (
        "Our team needs to establish a UctObservationObjective focused on unknown objects with UCT IDs 87654, 43210, and 86420. "
        "We'll employ sensor platforms TRACK05 and RADIO88 with S marking in REAL data collection mode. "
        "This operation should run in the MEO regime with a high priority of 4 and must complete 9 revisits hourly. "
        "Begin operations at 2025-07-19 00:00:00+00:00 with no defined end time. "
        "For image collection, configure 10 frames per observation with 3 second integration periods and binning level 2. "
        "Set end time offset to 90 minutes and ensure visibility checking is active. Enable sorting by brightest UCT."
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["87654", "43210", "86420"],
        "sensor_name_list": ["TRACK05", "RADIO88"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 9.0,
        "number_of_frames": 10,
        "integration_time": 3,
        "binning": 2,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 7, 19, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 4,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 87: SingleIntentObjective
    (
        "Would you prepare a SingleIntentObjective observation for target ID 34567 and corresponding RSO ID 98989? "
        "We need to deploy sensors AFB30 and STRC01 with TS classification under SIMULATED mode. "
        "Configure for SIDEREAL tracking with priority 6 and schedule the objective for 2025-08-25 13:30:00+00:00 ending at 2025-08-25 15:30:00+00:00. "
        "This requires 8 frames with 4.5 second integration time and binning level 3."
    ): {
        "classification_marking": "TS",
        "target_id": "34567",
        "rso_id": "98989",
        "sensor_name_list": ["AFB30", "STRC01"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 8,
        "integration_time": 4.5,
        "priority": 6,
        "binning": 3,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 8, 25, 13, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 25, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 88: DataEnrichmentObjective
    (
        "Establish a DataEnrichmentObjective focusing on targets 66778, 99887, and 33221. The operation should utilize "
        "sensor systems LMNT45 and RME77 with S//FOUO marking in EXERCISE mode. Configure the system to track up to 12 RSOs "
        "simultaneously with 15 revisits hourly. Schedule operations to commence on 2025-09-03 at 17:45:00+00:00 and continue "
        "until 2025-09-05 at 17:45:00+00:00. Set collection priority to 15 with RATE_TRACK approach. Disable visibility checking "
        "for this operation and utilize binning level 2."
    ): {
        "classification_marking": "S//FOUO",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["66778", "99887", "33221"],
        "sensor_name_list": ["LMNT45", "RME77"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "binning": 2,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 9, 3, 17, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 5, 17, 45, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 89: SensorCheckoutObjective
    (
        "Create a SensorCheckoutObjective to validate functionality of sensor STRC08. Apply C//FOUO classification under TEST conditions. "
        "Target the XGEO orbital regime using SIDEREAL tracking methodology. Set priority to 9 with 2 revisits per hour. "
        "Begin checkout procedure at 2025-10-12 05:30:00+00:00 and continue until 2025-10-12 11:30:00+00:00. "
        "Configure image capture for 15 frames per observation with 1.75 second integration and binning level 1. "
        "Set patience window to 25 minutes and deactivate visibility checking."
    ): {
        "classification_marking": "C//FOUO",
        "sensor_name": "STRC08",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 2.0,
        "number_of_frames": 15,
        "integration_time": 1.75,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 12, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 12, 11, 30, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 9,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 90: BaselineAutonomyObjective
    (
        "Initialize a BaselineAutonomyObjective with UUID 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'. "
        "Apply S classification with SIMULATED data collection mode. Set this as the highest priority task (priority value 500) "
        "to continuously track catalog IDs 29876 and 65432. Include additional RSO monitoring for IDs 11223, 44556, and 87654. "
        "Operation should run indefinitely with no end time and use LIGHT frame type."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "classification_marking": "S",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "priority": 500,
        "baseline_autonomy_rso": "29876,65432",
        "objective_end_time": None,
        "rso_id_list": ["11223", "44556", "87654"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 91: CatalogMaintenanceObjective
    (
        "Alright, let's get a CatalogMaintenanceObjective cookin'. We need it for sensors 'SKYEYE7' and 'WATCHTOWER3'. "
        "Slap a 'C' marking on it, and set the mode to 'SIMULATED'. How about a priority of 50? Patience can be, say, 15 minutes. "
        "And let's make the end time offset 30 minutes. Visibility check? Nah, keep that false. "
        "This whole thing should kick off on June 10, 2025, at 10:00:00 UTC and wrap up by June 10, 2025, 14:00:00 UTC. "
        "We're gonna use 'RATE_TRACK' for the tracking this time, and it's for the 'MEO' regime. "
        "Oh, and the RSO IDs are '98765' and '98766'."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 50,
        "rso_id_list": ["98765", "98766"],
        "sensor_name_list": ["SKYEYE7", "WATCHTOWER3"],
        "visibility_check": False,
    },
    # New Example 92: SearchObjective
    (
        "Okay, next up: a SearchObjective. We're looking for target 'TARGET_ALPHA_001' with the 'STARGAZR01' sensor. "
        "Let's go with 'S' marking and 'EXERCISE' mode. Priority should be pretty high, say 3. "
        "We'll stick with 'RATE_TRACK_SIDEREAL' tracking. It needs to start on July 15, 2025, at 08:00:00 UTC and finish by 10:30:00 UTC. "
        "Initial offset of 45 seconds, final offset 75 seconds. Frame overlap, hmm, 60% sounds good. "
        "The end time offset will be 35 minutes. Let's make the search type 'ALONG_TRACK'. "
        "Search starts 20 mins post objective start. Oh, and number of frames should be 10, with an integration time of 0.5 seconds."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 0.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 7, 15, 10, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 15, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 7, 15, 8, 20, 0, tzinfo=TzInfo(UTC))",
        "search_type": "ALONG_TRACK",
        "sensor_name": "STARGAZR01",
        "target_id": "TARGET_ALPHA_001",
        "visibility_check": False,
    },
    # New Example 93: GeodssRevisitObjective
    (
        "Time for a GeodssRevisitObjective. We need to hit targets 'GEO_SAT_X1' and 'GEO_SAT_Y2' using sensors 'GEOSCAN_A' and 'GEOSCAN_B'. "
        "Mark it 'U//FOUO', mode is 'REAL', priority 8. Tracking will be 'SIDEREAL'. Let's start this on August 20, 2025, at 23:00:00 UTC. "
        "For the GEODSS specific stuff: readout_rate 0 (that's 1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), "
        "auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), "
        "and scan_mode 0 (Continuous). Let's also set patience to 25 minutes and number of observations to 3."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 0,
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 20, 23, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 25,
        "priority": 8,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEOSCAN_A", "GEOSCAN_B"],
        "soi_filter_position": 2,
        "target_id_list": ["GEO_SAT_X1", "GEO_SAT_Y2"],
        "visibility_check": False,
    },
    # New Example 94: PeriodicRevisitObjective
    (
        "Let's whip up a PeriodicRevisitObjective. This one's for targets 'TGT_REVISIT_A' and 'TGT_REVISIT_B', "
        "using sensors 'PERSIST_1' and 'PERSIST_2'. Classification 'U', data mode 'TEST', and a low priority of 100. "
        "Patience can be the standard 30 minutes. Definitely ignore other objective submissions for this one. "
        "Start it on September 5, 2025, 05:30:00 UTC. We want an optimal 300 frames per hour, "
        "need 3 frames per intent, and an integration time of 1.5 seconds. Let's also specify revisits per hour as 4.0 and binning of 2."
    ): {
        "classification_marking": "U",
        "target_id_list": ["TGT_REVISIT_A", "TGT_REVISIT_B"],
        "sensor_name_list": ["PERSIST_1", "PERSIST_2"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 30,
        "revisits_per_hour": 4.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": 2,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 100,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 95: UctObservationObjective
    (
        "Next, a UctObservationObjective. We're tracking UCT RSOs 'UCT_OBJ_007' and 'UCT_OBJ_008' with sensors 'DEEPFIELD_X' and 'DEEPFIELD_Y'. "
        "This needs a 'TS' marking, 'REAL' mode, and it's for the 'XGEO' regime. Priority: a cool 7. Let's aim for 8.0 revisits per hour. "
        "Kick it off on October 10, 2025, 12:00:00 UTC. Yeah, sort by brightest UCT. End time offset should be 75 minutes. "
        "Visibility check is a must, so true for that. We need 4 frames with an integration time of 2.5 seconds. Set patience to 20 mins."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["UCT_OBJ_007", "UCT_OBJ_008"],
        "sensor_name_list": ["DEEPFIELD_X", "DEEPFIELD_Y"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 4,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 96: SingleIntentObjective
    (
        "Okay, let's set up a SingleIntentObjective. Target ID is 'SINGLE_TGT_Q5', RSO ID 'RSO_XYZ_123'. "
        "We'll use sensors 'PINPOINT_A' and 'FOCUS_B'. 'U//FOUO' for marking, 'SIMULATED' mode. Tracking type: 'RATE_TRACK'. "
        "Priority is 15. Let this one start on November 1, 2025, at 02:00:00 UTC. "
        "We need 2 frames, integration time 0.8 seconds, and binning of 4. "
        "Also, specify the intent start time as November 1, 2025, at 02:05:00 UTC and intent end time as November 1, 2025, at 02:15:00 UTC."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "SINGLE_TGT_Q5",
        "rso_id": "RSO_XYZ_123",
        "sensor_name_list": ["PINPOINT_A", "FOCUS_B"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 2,
        "integration_time": 0.8,
        "priority": 15,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2025, 11, 1, 2, 5, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 11, 1, 2, 15, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 2, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 97: DataEnrichmentObjective
    (
        "Now for a DataEnrichmentObjective. Let's enrich data for targets 'ENRICH_01', 'ENRICH_02', and 'ENRICH_03'. "
        "Use sensors 'DATASTREAM_7' and 'INFOCOLLECT_4'. Marking 'C', mode 'EXERCISE'. "
        "Critically, use 'RATE_TRACK' for tracking. We'll observe a max of 5 RSOs, with 15 revisits per hour. "
        "This objective should start on December 5, 2025, at 18:30:00 UTC. Visibility check should be on, so true. "
        "And let's set the priority to 25. Objective UUID should be 'enrich-uuid-001-test'."
    ): {
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "objective_uuid": "enrich-uuid-001-test",
        "target_id_list": ["ENRICH_01", "ENRICH_02", "ENRICH_03"],
        "sensor_name_list": ["DATASTREAM_7", "INFOCOLLECT_4"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 18, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 98: SensorCheckoutObjective
    (
        "Let's do a SensorCheckoutObjective. Classification 'U', sensor is 'CHECKMATE_SENSOR_1'. Mode 'REAL', and it's for the 'LEO' regime. "
        "'RATE_TRACK_SIDEREAL' is the tracking type. Priority will be 5. We want 2.5 revisits per hour. "
        "Start this on January 10, 2026, 09:00:00 UTC. Visibility check is true. Patience: 45 minutes. "
        "And we need 6 frames with 3 seconds of integration time. Use binning 1."
    ): {
        "classification_marking": "U",
        "sensor_name": "CHECKMATE_SENSOR_1",
        "orbital_regime": "LEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 2.5,
        "number_of_frames": 6,
        "integration_time": 3.0,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 99: BaselineAutonomyObjective
    (
        "Finally, a BaselineAutonomyObjective. Objective UUID is 'baseline-autonomy-live-007'. "
        "Markings are 'S', mode is 'REAL'. Frame type can be 'DARK' for this one. Priority: the usual 1000 for baseline. "
        "The baseline autonomy RSOs are catalog IDs 'SATCAT001,SATCAT002,SATCAT003'. "
        "This should run continuously, so no end time. The RSO ID list includes 'RSO_BASE_1', 'RSO_BASE_2', and 'RSO_BASE_3', "
        "though we know this gets overwritten."
    ): {
        "objective_uuid": "baseline-autonomy-live-007",
        "classification_marking": "S",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 1000,
        "baseline_autonomy_rso": "SATCAT001,SATCAT002,SATCAT003",
        "objective_end_time": None,
        "rso_id_list": ["RSO_BASE_1", "RSO_BASE_2", "RSO_BASE_3"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 100: CatalogMaintenanceObjective
    (
        "Hey there! I need to set up a CatalogMaintenanceObjective using sensors LMNT08 and RME19 with TS classification markings. Can you make it run in TEST mode with a priority of 7? Oh, and it should have a patience time of 15 minutes and end time offset of 30 minutes. Make sure visibility check is enabled. I want it to start tomorrow at 2025-05-16 08:45:00+00:00 and end at 2025-05-16 14:30:00+00:00. Let's use RATE_TRACK_SIDEREAL tracking in GEO regime. The RSO ID list needs to include '33421,98765,44556'."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 5, 16, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 16, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 15,
        "priority": 7,
        "rso_id_list": ["33421", "98765", "44556"],
        "sensor_name_list": ["LMNT08", "RME19"],
        "visibility_check": True,
    },
    # New Example 101: SearchObjective
    (
        "Could you set up a SearchObjective for target 78901 with the UKR33 sensor? It needs C classification marking and REAL mode. Set priority to 8 and use SIDEREAL tracking. Start time should be 2025-05-20 13:45:00+00:00, ending at 2025-05-20 16:15:00+00:00. I want initial offset of 45 seconds and final offset of 75 seconds, with 60% frame overlap. Make the end time offset 35 minutes. And yeah, let's do CROSS_TRACK search type with search start time 20 minutes after objective start."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 5, 20, 16, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 20, 13, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 8,
        "search_start_time": "datetime.datetime(2025, 5, 20, 14, 5, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "UKR33",
        "target_id": "78901",
        "visibility_check": False,
    },
    # New Example 102: GeodssRevisitObjective
    (
        "Yo, I need a GeodssRevisitObjective for targets 23456,78910,11213 using RME28 and LMNT31 sensors. Make it S//FOUO classification in TEST mode with priority 9. Let's use RATE_TRACK tracking. Start on 2025-05-18 at 03:15:00+00:00. Set readout_rate to 0 (1MHz), gain_setting to 1 (Low Gain), soi_filter_position to 2 (10% Light), auto_track_type to 2 (Manual), camera_mode to 1 (Zoomed EBS), and array_kind to 0 (Main). Use binning_mode 0 (No Binning) and scan_mode 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 0,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "S//FOUO",
        "collect_request_type": "RATE_TRACK",
        "command": 0,
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 18, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "patience_minutes": 30,
        "priority": 9,
        "readout_rate_setting": 0,
        "scan_mode": 0,
        "sensor_name_list": ["RME28", "LMNT31"],
        "soi_filter_position": 2,
        "target_id_list": ["23456", "78910", "11213"],
        "visibility_check": False,
    },
    # New Example 103: PeriodicRevisitObjective
    (
        "Set up a PeriodicRevisitObjective for targets 76543,98765 with RME07 and LMNT09 sensors. Classification should be U//FOUO, use REAL mode, priority 3, patience of 45 mins, and don't ignore other objective submissions. Start this thing at 2025-06-01 22:00:00+00:00. Set optimal frames per hour to 350, number of frames to 8, and integration time to 3.5 seconds. Make sure visibility check is turned on."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["76543", "98765"],
        "sensor_name_list": ["RME07", "LMNT09"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "number_of_frames": 8,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 6, 1, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 3,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 104: UctObservationObjective
    (
        "Listen, I need a UctObservationObjective to track UCT RSOs 54321,88776,99001 with sensors RME25 and LMNT26. Classification is TS, REAL mode, MEO regime, priority 5, with 8 revisits per hour. We'll start it on 2025-05-30 at 14:30:00+00:00. Make end time offset 90 minutes and turn on visibility checking. Don't sort by brightest UCT. Set number of frames to 7 and integration time to 4 seconds. Oh, and use binning level 2."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["54321", "88776", "99001"],
        "sensor_name_list": ["RME25", "LMNT26"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 4,
        "binning": 2,
        "end_time_offset_minutes": 90,
        "objective_start_time": "datetime.datetime(2025, 5, 30, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 105: SingleIntentObjective
    (
        "Can you create a SingleIntentObjective with target ID 87654 and RSO ID 12345? Let's use the RME41 and LMNT42 sensors. Make it U//FOUO classification in SIMULATED mode with SIDEREAL tracking. Priority should be 6. Start the objective at 2025-06-05 08:00:00+00:00 and set an end time at 2025-06-05 10:00:00+00:00. Use 6 frames with 1.5 second integration time and binning level 4."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "87654",
        "rso_id": "12345",
        "sensor_name_list": ["RME41", "LMNT42"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 6,
        "integration_time": 1.5,
        "priority": 6,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 6, 5, 8, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 5, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 106: DataEnrichmentObjective
    (
        "Let's set up a DataEnrichmentObjective for targets 12121, 34343, 56565 using sensors RME11 and LMNT12. Use C classification, EXERCISE mode, and SIDEREAL tracking. Maximum RSOs to observe should be 10, with 15 revisits per hour. Start at 2025-07-10 16:30:00+00:00 and end at 2025-07-11 16:30:00+00:00. Set priority to 15 and enable visibility checking."
    ): {
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "target_id_list": ["12121", "34343", "56565"],
        "sensor_name_list": ["RME11", "LMNT12"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 7, 10, 16, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 7, 11, 16, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 107: SensorCheckoutObjective
    (
        "I need a SensorCheckoutObjective for sensor RME50. Classification is S, data mode is TEST, orbital regime is LEO, and collect request type is RATE_TRACK. Give it priority 12 with 2.5 revisits per hour. Start tomorrow at 2025-05-16 12:00:00+00:00. Set visibility check to false, patience to 60 minutes, use 10 frames with 5 second integration time, and binning level 1."
    ): {
        "classification_marking": "S",
        "sensor_name": "RME50",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 60,
        "revisits_per_hour": 2.5,
        "number_of_frames": 10,
        "integration_time": 5,
        "binning": 1,
        "objective_start_time": "datetime.datetime(2025, 5, 16, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 12,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 108: BaselineAutonomyObjective
    (
        "Create a BaselineAutonomyObjective with UUID '987f6543-d21c-34a5-b678-912345678901'. We need S//FOUO classification, EXERCISE mode, LIGHT frame type, and priority 800. Include these RSO IDs: 33333, 44444, 55555, and these catalog IDs: 28350, 39461, 44192. Set an end time of 2025-06-30 23:59:59+00:00."
    ): {
        "objective_uuid": "987f6543-d21c-34a5-b678-912345678901",
        "classification_marking": "S//FOUO",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "priority": 800,
        "baseline_autonomy_rso": "28350,39461,44192",
        "objective_end_time": "datetime.datetime(2025, 6, 30, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["33333", "44444", "55555"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 109: CatalogMaintenanceObjective
    (
        "Oh, joy, another CatalogMaintenanceObjective. Fine. Make it for sensors XYZ01 and ABC02. Use 'S' markings, because apparently, that's a thing. "
        "Set it to REAL mode, priority 50 because it's *so* important. Give it a patience of, I don't know, 15 minutes? And an end time offset of 30 minutes, whatever that means. "
        "Make sure visibility check is true, because why not? Start this thrilling objective on 2025-10-10 at 10:00:00+00:00 and let it mercifully end on 2025-10-10 at 12:00:00+00:00. "
        "Oh, and track it with RATE_TRACK in the MEO regime. The RSO IDs are '54321' and '98760'. Don't forget the binning, set it to 2. Is that enough detail for you?"
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["XYZ01", "ABC02"],
        "rso_id_list": ["54321", "98760"],
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": 2,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 110: SearchObjective
    (
        "Alright, genius, conjure up a SearchObjective. We're looking for target 'target_alpha_7' with sensor 'SENSRX9'. Use 'TS' markings - super secret stuff, obviously. "
        "Mode is 'SIMULATED' because we're just playing make-believe. Priority is a stunning 3. Use 'SIDEREAL' tracking, because 'RATE_TRACK_SIDEREAL' is just too mainstream. "
        "Objective starts 2025-11-01 at 08:00:00+00:00 and, if we're lucky, ends 2025-11-01 at 09:30:00+00:00. "
        "Set initial offset to 45 seconds, final offset to 75 seconds. Frame overlap should be a nice, round 0.6 (or 60%, if you prefer). End time offset is 30 minutes. "
        "And the search type is 'CROSS_TRACK'. Oh, and let the `search_start_time` be 10 minutes after the objective begins, just to keep things spicy. And visibility check false, because we like to live dangerously."
    ): {
        "classification_marking": "TS",
        "target_id": "target_alpha_7",
        "sensor_name": "SENSRX9",
        "search_type": "CROSS_TRACK",
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 30,
        "priority": 3,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 1, 9, 30, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": None,
        "integration_time": None,
        "search_start_time": "datetime.datetime(2025, 11, 1, 8, 10, 0, tzinfo=TzInfo(UTC))",  # 10 mins after objective_start_time
        "objective_name": "SearchObjective",
    },
    # New Example 111: GeodssRevisitObjective
    (
        "Ugh, a GeodssRevisitObjective. Aren't these just *fascinating*? For targets 'geo_target_1' and 'geo_target_2', using sensors 'GEOSNS007' and 'GEOSNS008'. "
        "Marking is 'U//FOUO', naturally. Mode is 'EXERCISE'. Priority, let's say 15. Tracking type is 'RATE_TRACK_SIDEREAL', the usual. Start it on 2025-12-05 at 20:00:00+00:00. "
        "Now for the *really* exciting part: readout_rate 0 (that's 1MHz, thrilling), gain_setting 1 (Low Gain, riveting), soi_filter 2 (10% Light, groundbreaking stuff), "
        "auto_track_type 2 (Manual, because automatic is for rookies), camera_mode 1 (Zoomed EBS, wow), array_kind 1 (Photometer, truly sensational), "
        "binning_mode 0 (No Binning, how avant-garde), and scan_mode 0 (Continuous, because why stop?). Set number of frames to 10 and integration time to 0.5 seconds. And `patience_minutes` can be 25. Try not to mess this one up."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["geo_target_1", "geo_target_2"],
        "sensor_name_list": ["GEOSNS007", "GEOSNS008"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default as not specified
        "patience_minutes": 25,
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 0.5,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 15,
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
    # New Example 112: PeriodicRevisitObjective
    (
        "Okay, let's get this over with. A PeriodicRevisitObjective. For targets 'periodic_A' and 'periodic_B', using sensors 'SENSOR_P1' and 'SENSOR_P2'. "
        "'C' marking, 'TEST' mode, priority a mere 7. Patience is 40 minutes. We want, oh, 5 revisits per hour. Let's say 3 frames per intent, with an integration time of 1.5 seconds. "
        "Start this thrilling endeavor on 2026-01-15 at 00:00:00+00:00 and let it run until 2026-01-15 06:00:00+00:00. "
        "Set `ignore_other_objective_intent_submissions` to true, because we don't care about others. And optimal frames per hour? A cool 300. Visibility check should be true. And `binning` to 1. Can you handle that?"
    ): {
        "classification_marking": "C",
        "target_id_list": ["periodic_A", "periodic_B"],
        "sensor_name_list": ["SENSOR_P1", "SENSOR_P2"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 5.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": 1,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 15, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 113: UctObservationObjective
    (
        "Time for a UctObservationObjective. This sounds important, doesn't it? For UCT RSOs 'uct_rso_X' and 'uct_rso_Y', with sensors 'UCT_SENSOR_1' and 'UCT_SENSOR_2'. "
        "'U//FOUO' marking, 'REAL' mode, in the 'XGEO' regime because we're ambitious. Priority 8. We need 4.5 revisits per hour. "
        "Set 6 frames, integration time of 2.5 seconds. Start it on 2026-02-20 at 12:00:00+00:00. End time offset is, say, 75 minutes. "
        "And yes, please sort by brightest UCT, because we only want the shiny ones. Visibility check on, `patience_minutes` 35. Oh, and `binning` can be `None` for this one, just to be different."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["uct_rso_X", "uct_rso_Y"],
        "sensor_name_list": ["UCT_SENSOR_1", "UCT_SENSOR_2"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 35,
        "revisits_per_hour": 4.5,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 114: SingleIntentObjective
    (
        "Just a SingleIntentObjective now. How quaint. Target ID 'single_target_Q', RSO ID 'single_rso_99'. Use sensors 'SINGLE_SENS_A' and 'SINGLE_SENS_B'. "
        "'U' marking, 'SIMULATED' mode. Collect type 'RATE_TRACK'. Priority a solid 12. Let's go with 7 frames, integration time of 3 seconds, and binning of 4. "
        "The objective should start on 2026-03-10 at 15:30:00+00:00. No specific end time for the objective, let it run its course, or rather, let the intent timings dictate. "
        "Set `intent_start_time` for 2026-03-10 15:35:00+00:00 and `intent_end_time` for 2026-03-10 15:45:00+00:00. Simple, right? Even for you."
    ): {
        "classification_marking": "U",
        "target_id": "single_target_Q",
        "rso_id": "single_rso_99",
        "sensor_name_list": ["SINGLE_SENS_A", "SINGLE_SENS_B"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 7,
        "integration_time": 3.0,
        "priority": 12,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2026, 3, 10, 15, 35, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 3, 10, 15, 45, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 115: DataEnrichmentObjective
    (
        "Let's enrich some data, shall we? With a DataEnrichmentObjective. Targets 'data_E_1', 'data_E_2', 'data_E_3'. Sensors 'ENRICH_01', 'ENRICH_02'. "
        "'S' marking, 'REAL' mode. Tracking is 'SIDEREAL'. Max RSO to observe is 7. We demand 15 revisits per hour. "
        "Start this on 2026-04-05 at 09:00:00+00:00 and end it at 2026-04-05 17:00:00+00:00. Priority 25. Visibility check must be true. "
        "And for fun, set `binning` to 3. `intent_start_time` can be 5 minutes after objective start, and `intent_end_time` 5 minutes before objective end. Don't bore me."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["data_E_1", "data_E_2", "data_E_3"],
        "sensor_name_list": ["ENRICH_01", "ENRICH_02"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": 3,
        "max_rso_to_observe": 7,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 4, 5, 17, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "intent_start_time": "datetime.datetime(2026, 4, 5, 9, 5, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 4, 5, 16, 55, 0, tzinfo=TzInfo(UTC))",
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 116: SensorCheckoutObjective
    (
        "Time to check out a sensor. SensorCheckoutObjective for 'SENSOR_CHK_001'. 'U' marking, in the 'LEO' regime. 'TEST' mode. "
        "Collect type 'RATE_TRACK_SIDEREAL'. Let's have 2 revisits per hour. 4 frames, 1 second integration time. "
        "Start on 2026-05-01 at 07:00:00+00:00. End it on 2026-05-01 10:00:00+00:00. Priority 11. Visibility check on, patience 20 minutes. "
        "No binning for this, `None` will do. And make sure the `intent_start_time` is 10 mins after objective start, and `intent_end_time` is 10 mins before objective end. Try to keep up."
    ): {
        "classification_marking": "U",
        "sensor_name": "SENSOR_CHK_001",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 2.0,
        "number_of_frames": 4,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 7, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 5, 1, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": "datetime.datetime(2026, 5, 1, 7, 10, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 5, 1, 9, 50, 0, tzinfo=TzInfo(UTC))",
        "priority": 11,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 117: BaselineAutonomyObjective
    (
        "Finally, the BaselineAutonomyObjective. Give it UUID 'abcdef01-2345-6789-abcd-ef0123456789'. 'U//FOUO' markings, 'REAL' mode. "
        "Frame type 'DARK' just to be difficult. Priority an absurdly high 500. The `baseline_autonomy_rso` should be 'catID_A,catID_B,catID_C'. "
        "And the `rso_id_list` should include 'rso_base_1' and 'rso_base_2'. Let this one run forever, so no `objective_end_time`. Get on with it, this is the last one, thank goodness."
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 500,
        "baseline_autonomy_rso": "catID_A,catID_B,catID_C",
        "objective_end_time": None,
        "rso_id_list": ["rso_base_1", "rso_base_2"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 118: CatalogMaintenanceObjective
    (
        "For the love of all things orbital, set up a CatalogMaintenanceObjective with S classification marking. Use sensor ZTF09 because apparently that's the only one that works these days. Set MEO regime, patience of 15 minutes (like we have all day to wait), and an end time offset of 40 minutes. Track with RATE_TRACK_SIDEREAL obviously, because who would want anything else? Use TEST mode so when it inevitably fails, nobody important notices. Start tomorrow at 2025-05-16 14:30:00+00:00 and run until 2025-05-16 23:45:00+00:00. Priority 8, because it's not like we have anything better to do. Add RSO IDs '34521' and '98712' to the list if you can manage to do that correctly."
    ): {
        "classification_marking": "S",
        "sensor_name_list": ["ZTF09"],
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 40,
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "objective_start_time": "datetime.datetime(2025, 5, 16, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 16, 23, 45, tzinfo=TzInfo(UTC))",
        "priority": 8,
        "rso_id_list": ["34521", "98712"],
        "objective_name": "CatalogMaintenanceObjective",
        "frame_type": "LIGHT",
        "visibility_check": False,
    },
    # New Example 119: SearchObjective
    (
        "Ugh, fine, I'll set up a SearchObjective for target ID 78910 using the UKR43 sensor, which is probably offline anyway. Use U//FOUO marking because this is sooo important (eye roll). Set REAL mode - yes, we're actually doing this. Priority 3 because someone upstairs thinks this is urgent. Set initial offset to 45 seconds and final offset to 120 seconds. Frame overlap should be 65% - don't mess that up. End time offset 30 minutes. Start at 2025-05-22 08:15:00+00:00 and end at 2025-05-22 10:45:00+00:00. Search start time should be 2025-05-22 08:30:00+00:00. Oh, and use CROSS_TRACK search type for once, just to mix things up a bit."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "78910",
        "sensor_name": "UKR43",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "initial_offset": 45,
        "final_offset": 120,
        "frame_overlap_percentage": 0.65,
        "end_time_offset_minutes": 30,
        "objective_start_time": "datetime.datetime(2025, 5, 22, 8, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 22, 10, 45, tzinfo=TzInfo(UTC))",
        "priority": 3,
        "search_type": "CROSS_TRACK",
        "search_start_time": "datetime.datetime(2025, 5, 22, 8, 30, tzinfo=TzInfo(UTC))",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "objective_name": "SearchObjective",
    },
    # New Example 120: GeodssRevisitObjective
    (
        "I need a GeodssRevisitObjective for targets 55667 and 44332 - yes, both of them, try to keep up. Use sensors LMNT56 and RME77 if they're not busy doing something more important. C classification, because we're feeling fancy today. Set REAL mode, and stick with RATE_TRACK_SIDEREAL tracking. Priority 15 - not urgent, but the boss is watching. Start on 2025-06-10 22:00:00+00:00. Set readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 0 (No Autotrack), camera_mode 1 (Zoomed EBS), array_kind 0 (Main), binning_mode 0 (No Binning), scan_mode 0 (Continuous). Surely you can handle that without asking me twenty follow-up questions."
    ): {
        "classification_marking": "C",
        "target_id_list": ["55667", "44332"],
        "sensor_name_list": ["LMNT56", "RME77"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 22, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "soi_filter_position": 2,
        "auto_track_type": 0,
        "camera_mode": 1,
        "array_kind": 0,
        "binning_mode": 0,
        "scan_mode": 0,
        "frame_type": "LIGHT",
        "patience_minutes": 30,
        "visibility_check": False,
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 121: PeriodicRevisitObjective
    (
        "Create a PeriodicRevisitObjective for targets 55123 and 89456, if they're even still in orbit. Use sensors RME11 and LMNT22 - they're the least likely to malfunction based on our stellar maintenance record. Classification should be TS because someone's paranoid again. REAL mode, obviously, because simulation is for the weak. Priority 7, patience 45 minutes (we'll need it), and revisits per hour at 2.5 (ambitious, I know). Start at 2025-07-05 03:45:00+00:00 and set number of frames to 8 with integration time of 3.5 seconds. Oh, and turn on visibility check for once - let's not waste time pointing at empty sky."
    ): {
        "classification_marking": "TS",
        "target_id_list": ["55123", "89456"],
        "sensor_name_list": ["RME11", "LMNT22"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 2.5,
        "number_of_frames": 8,
        "integration_time": 3.5,
        "objective_start_time": "datetime.datetime(2025, 7, 5, 3, 45, tzinfo=TzInfo(UTC))",
        "priority": 7,
        "frame_type": "LIGHT",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 122: UctObservationObjective
    (
        "Listen carefully because I'm only explaining this once. Set up a UctObservationObjective for UCT RSOs 12378 and 45699 using sensors RME25 and LMNT37. Classification is U - nothing exciting to see here. GEO regime, REAL mode, and RATE_TRACK_SIDEREAL tracking as usual. Let's do 4 revisits per hour since we apparently have nothing better to do with our telescope time. Start at 2025-08-18 16:30:00+00:00, priority 8, with end time offset of 90 minutes. Number of frames should be 6, integration time 2.5 seconds. And yes, sort by brightest UCT first - at least try to find something visible. Visibility check enabled, obviously, unless you enjoy staring at clouds."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["12378", "45699"],
        "sensor_name_list": ["RME25", "LMNT37"],
        "orbital_regime": "GEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "revisits_per_hour": 4.0,
        "objective_start_time": "datetime.datetime(2025, 8, 18, 16, 30, tzinfo=TzInfo(UTC))",
        "priority": 8,
        "end_time_offset_minutes": 90,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "sort_by_brightest_uct": True,
        "visibility_check": True,
        "frame_type": "LIGHT",
        "patience_minutes": 30,
        "objective_name": "UctObservationObjective",
    },
    # New Example 123: SingleIntentObjective
    (
        "Hurry up and create a SingleIntentObjective with target ID 12355 and RSO ID 87643. Use sensors RME30 and LMNT35 - assuming they've been calibrated this century. Classification U//FOUO because somebody thinks this is special. REAL mode with SIDEREAL tracking for a change. Set priority to 6, because 5 would be too important and 7 would be too trivial. Start at 2025-09-03 05:15:00+00:00. Take 10 frames with 1.5 second integration time. Binning should be 3, which is probably overkill but whatever the request form says."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "12355",
        "rso_id": "87643",
        "sensor_name_list": ["RME30", "LMNT35"],
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "number_of_frames": 10,
        "integration_time": 1.5,
        "priority": 6,
        "binning": 3,
        "objective_start_time": "datetime.datetime(2025, 9, 3, 5, 15, tzinfo=TzInfo(UTC))",
        "frame_type": "LIGHT",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 124: DataEnrichmentObjective
    (
        "Fine, let's set up another DataEnrichmentObjective. Targets are 77889, 11223, and 44556 - which I had to triple-check because someone can't write legibly. Use sensors RME66 and LMNT71 if they're functioning today. S classification, TEST mode (nobody trusts this to work in REAL). Set RATE_TRACK instead of the default - I know, shocking change. Set max RSO to observe to 12 and 8 revisits per hour, because apparently we have unlimited power and tracking capacity. Start at 2025-10-11 12:00:00+00:00. Visibility check enabled, and priority 15 because it's definitely not urgent."
    ): {
        "classification_marking": "S",
        "data_mode": "TEST",
        "target_id_list": ["77889", "11223", "44556"],
        "sensor_name_list": ["RME66", "LMNT71"],
        "collect_request_type": "RATE_TRACK",
        "max_rso_to_observe": 12,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2025, 10, 11, 12, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "visibility_check": True,
        "frame_type": "LIGHT",
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 125: SensorCheckoutObjective
    (
        "Great, now I have to babysit a sensor checkout. Create a SensorCheckoutObjective for RME88 - you know, the one that's been malfunctioning all month. Classification U because nobody cares. Set LEO regime instead of the usual GEO, just to be difficult. REAL mode, with RATE_TRACK tracking. Priority 12, because sensor checkouts are just sooo important. Set 3 revisits per hour, patience of 40 minutes, and number of frames to 4 with integration time of 1.8 seconds. Start tomorrow at 2025-05-16 10:00:00+00:00, and yes, enable visibility check - we've wasted enough time already pointing at nothing."
    ): {
        "classification_marking": "U",
        "sensor_name": "RME88",
        "orbital_regime": "LEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "priority": 12,
        "revisits_per_hour": 3.0,
        "patience_minutes": 40,
        "number_of_frames": 4,
        "integration_time": 1.8,
        "objective_start_time": "datetime.datetime(2025, 5, 16, 10, 0, tzinfo=TzInfo(UTC))",
        "visibility_check": True,
        "frame_type": "LIGHT",
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 126: BaselineAutonomyObjective
    (
        "Ugh, I have to set up another BaselineAutonomyObjective with UUID '456e7890-f12a-34bc-d567-890123456789'. Set U marking - nothing classified happening here, unfortunately. TEST mode because we don't trust this to run in production yet. Make it priority 800 - not quite the lowest but close enough. Tracking catalog IDs are '43281' and '76592', and while you're at it, add RSO ids '11111', '22222', and '33333'. No end time because apparently this needs to run forever until someone remembers to turn it off."
    ): {
        "objective_uuid": "456e7890-f12a-34bc-d567-890123456789",
        "classification_marking": "U",
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "priority": 800,
        "baseline_autonomy_rso": "43281,76592",
        "objective_end_time": None,
        "rso_id_list": ["11111", "22222", "33333"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 127: CatalogMaintenanceObjective
    (
        "Alright, team, listen up! We need to whip up a CatalogMaintenanceObjective. This one's for our trusty sensors 'WACKY01' and 'GOOFY02'. "
        "Let's slap a 'C' for 'Classifiedly Comical' marking on it. Set the mode to SIMULATED because, frankly, reality is overrated today. "
        "Priority? Oh, let's make it a saucy 69. We're feeling patient, so give it 45 minutes of patience, and an end time offset of a neat 30 minutes. "
        "Visibility check? Nah, we trust our gut - set it to false. "
        "This cosmic ballet begins on 2025-11-01 at 10:00:00+00:00 and the grand finale is on 2025-11-01 at 15:30:00+00:00. "
        "We're tracking these things in the MEO regime, using the ever-reliable RATE_TRACK. "
        "And the RSO IDs for this shindig are '11223' and '44556'. Oh, and let's try binning at 2x2 for extra chunky pixels!"
    ): {
        "binning": 2,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",  # Default
        "objective_end_time": "datetime.datetime(2025, 11, 1, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 11, 1, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 69,
        "rso_id_list": ["11223", "44556"],
        "sensor_name_list": ["WACKY01", "GOOFY02"],
        "visibility_check": False,
    },
    # New Example 128: SearchObjective
    (
        "Hark! A SearchObjective is what we require! Our elusive target is 'TARGET_ZORP', and the mighty sensor 'FINDR09' is on the case. "
        "This mission is classified 'S' for 'Seriously Searching Something'. We're in REAL mode, because this ain't no drill, people! "
        "Tracking type: SIDEREAL, nice and steady. Let's set the priority to a low and humble 3, because we're not too pushy. "
        "Commence the search operations on 2026-01-15 at 08:00:00+00:00, and don't you dare stop until 2026-01-15 at 12:30:00+00:00. "
        "Give us an initial offset of 120 seconds - we like to sneak up. Final offset will be 150 seconds, for a dramatic exit. "
        "Frame overlap should be a cozy 60% (that's 0.6 for you mathematicians), and the end time offset is a generous 50 minutes. "
        "We're going ALONG_TRACK for this hunt, and the specific search start time is precisely 20 minutes after the objective kicks off. No binning, keep it clean!"
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 50,
        "final_offset": 150,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",  # Default
        "initial_offset": 120,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2026, 1, 15, 12, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2026, 1, 15, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2026, 1, 15, 8, 20, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 20 minutes
        "search_type": "ALONG_TRACK",
        "sensor_name": "FINDR09",
        "target_id": "TARGET_ZORP",
        "visibility_check": False,  # Default
    },
    # New Example 129: GeodssRevisitObjective
    (
        "Hear ye, hear ye! A GeodssRevisitObjective is decreed! We're targeting 'ASTEROID_BERT' and 'COMET_ERNIE' with sensors 'GEOSNAP01' and 'STARGAZE77'. "
        "This is top-secret stuff, so mark it 'TS'. Mode is REAL, of course. Priority is a respectable 8. "
        "Let this grand observation commence on 2025-12-12 at 20:00:00+00:00, with no specific end time, let it run free! "
        "We want that sweet 1MHz readout_rate (that's value 0), and Low Gain (value 1) for sensitivity. "
        "Filter? Let's go with 10% Light (value 2). Auto_track_type is Manual (value 2), because we're hands-on. "
        "Camera_mode? Zoomed EBS, obviously (value 1). Array_kind will be Photometer (value 1). "
        "No Binning (value 0) for this one, and Scan_mode is Continuous (value 0). Patience can be the standard 30 minutes."
    ): {
        "acquisition_type": 0,  # Default
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "command": 0,  # Default
        "data_mode": "REAL",
        "frame_type": "LIGHT",  # Default
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 12, 12, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 30,  # Default
        "priority": 8,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEOSNAP01", "STARGAZE77"],
        "soi_filter_position": 2,
        "target_id_list": ["ASTEROID_BERT", "COMET_ERNIE"],
        "visibility_check": False,  # Default
    },
    # New Example 130: PeriodicRevisitObjective
    (
        "Gadzooks! It's time for a PeriodicRevisitObjective! Our VIP targets are 'SAT_ALPHA', 'SAT_BETA', and 'SAT_GAMMA'. "
        "Assign sensors 'PEEKABOO03' and 'LOOKYLOO04' to this task. Marking: 'U//FOUO', because it's Unclassified For Our Unusually Odd Observers. "
        "Mode is TEST, as we're just trying things out, you know, for giggles. Priority is a zesty 7. "
        "Patience can be a swift 15 minutes. We want to ignore other objective submissions, because we're divas: set that to true. "
        "Kick this off on 2026-02-10 at 05:30:00+00:00. We demand 3 revisits per hour! How about 10 frames per intent, with an integration time of 1.5 seconds? Sounds delightful. "
        "Optimal frames per hour? Let's go wild with 500!"
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["SAT_ALPHA", "SAT_BETA", "SAT_GAMMA"],
        "sensor_name_list": ["PEEKABOO03", "LOOKYLOO04"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 15,
        "revisits_per_hour": 3.0,
        "number_of_frames": 10,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 2, 10, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 131: UctObservationObjective
    (
        "Crikey! We've got some Unidentified Celestial Thingamajigs to eyeball! This calls for a UctObservationObjective! "
        "Our UCT RSO list includes 'UCT_SPARKY' and 'UCT_BLINKY'. Sensors on deck are 'MYSTERY01' and 'PROBER02'. "
        "Marking is 'U' for 'Unbelievably Unidentified'. Mode: REAL. We're looking at things in the XGEO regime, way out there! "
        "Priority 11, just because. We want, oh, 4.5 revisits per hour. This cosmic stakeout starts 2025-10-10 at 23:00:00+00:00. "
        "Let's make sure visibility check is true, we don't want to waste time. And yes, sort by the brightest UCT, let the shiny ones come first! "
        "Give us 8 frames with 3 seconds of integration time. End time offset can be the default 60 minutes."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["UCT_SPARKY", "UCT_BLINKY"],
        "sensor_name_list": ["MYSTERY01", "PROBER02"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 4.5,
        "number_of_frames": 8,
        "integration_time": 3.0,
        "binning": None,
        "end_time_offset_minutes": 60,  # Default
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 23, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 11,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 132: SingleIntentObjective
    (
        "Psst! Need a SingleIntentObjective, on the double! Target ID is 'GHOST_SAT_7', and its RSO ID is 'SPECTER_RSO_007'. "
        "Sensors 'EAGLEEYE_X' and 'NIGHTOWL_Z' are your chosen ones. This is 'C' for 'Covertly Captured'. "
        "Data mode is EXERCISE, we're practicing our ninja skills. Use RATE_TRACK tracking. Priority is a cool 5. "
        "Objective starts on 2026-03-03 at 03:03:03+00:00. Let's get 3 frames, integration time of 0.5 seconds, and binning of 4x4. "
        "No specific intent start or end time, let the system figure it out based on the objective start. This is a quick in-and-out job."
    ): {
        "classification_marking": "C",
        "target_id": "GHOST_SAT_7",
        "rso_id": "SPECTER_RSO_007",
        "sensor_name_list": ["EAGLEEYE_X", "NIGHTOWL_Z"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 3,
        "integration_time": 0.5,
        "priority": 5,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 3, 3, 3, 3, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 133: DataEnrichmentObjective
    (
        "More data! We need a DataEnrichmentObjective! Our targets for this data feast are 'SHINY_OBJECT_1', 'DULL_OBJECT_2', and 'BLINKY_OBJECT_3'. "
        "Use sensors 'DATASCOOP_A' and 'INFOVAC_B'. Marking: 'S'. Data mode: REAL. "
        "We're going with SIDEREAL tracking to gather that sweet, sweet data. Max RSOs to observe? Let's say 5, don't want to get greedy. "
        "We want a whopping 15 revisits per hour! Start this glorious data harvest on 2025-11-20 at 14:00:00+00:00. "
        "Priority is 22. And yes, visibility check is true; we only want to see what's seeable. No binning, please, we want raw, unadulterated pixels."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["SHINY_OBJECT_1", "DULL_OBJECT_2", "BLINKY_OBJECT_3"],
        "sensor_name_list": ["DATASCOOP_A", "INFOVAC_B"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 11, 20, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 22,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 134: SensorCheckoutObjective
    (
        "Time for a spa day for our sensors! Create a SensorCheckoutObjective for sensor 'LAZYBEAM007'. "
        "This is a 'U'nclassified checkout. Orbital regime is LEO, we're keeping it close to home. "
        "Data mode: SIMULATED. Collect type RATE_TRACK_SIDEREAL as per usual. Priority: a high and mighty 2, this is important! "
        "We want 0.5 revisits per hour, nice and slow. Start this checkout on 2026-04-01 (no foolin'!) at 09:00:00+00:00. "
        "Visibility check is false for this one. Patience of 60 minutes, we're very understanding. "
        "Let's grab 7 frames with an integration time of 4 seconds. Binning can be 1 (no binning, effectively)."
    ): {
        "classification_marking": "U",
        "sensor_name": "LAZYBEAM007",
        "orbital_regime": "LEO",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,
        "patience_minutes": 60,
        "revisits_per_hour": 0.5,
        "number_of_frames": 7,
        "integration_time": 4.0,
        "binning": 1,  # Assuming 1 might mean 1x1 or no effective binning
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 1, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 2,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 135: BaselineAutonomyObjective
    (
        "Behold! The grand BaselineAutonomyObjective! Assign it the majestic UUID 'abcdef01-2345-6789-abcd-ef0123456789'. "
        "Markings are 'U//FOUO'. Data mode is REAL. Frame type can be DARK for a change, let's see what the void looks like. "
        "Priority is a super low 2000, this thing just chugs along in the background. "
        "Our baseline autonomy RSOs, by their catalog IDs, are 'CATID_001,CATID_002,CATID_SUPERSTAR'. "
        "We want this to run until the end of time, or at least until 2027-01-01 at 00:00:00+00:00 because forever is a long time to code. "
        "The RSO ID list for dynamic updates can initially include 'RSO_TEMP_A' and 'RSO_TEMP_B'."
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 2000,
        "baseline_autonomy_rso": "CATID_001,CATID_002,CATID_SUPERSTAR",
        "objective_end_time": "datetime.datetime(2027, 1, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["RSO_TEMP_A", "RSO_TEMP_B"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 136: CatalogMaintenanceObjective
    (
        "Hey space cadets! I need to whip up a CatalogMaintenanceObjective for sensors DME77 and STLT99 - these babies need U//FOUO classification, please! We're running in REAL mode (no simulation shenanigans today) with a pretty chill priority of 15. The sensors need some patience, let's say 20 mins, and an end time offset of 35 mins would be fantabulous. Oh! Almost forgot - we absolutely MUST do a visibility check this time. Start this cosmic adventure on 2025-01-15 13:45:00+00:00 and wrap it up by 2025-01-15 16:15:00+00:00. Let's stick with RATE_TRACK_SIDEREAL tracking in the MEO regime. Last thing - we're watching RSO IDs '22233,44455,66677'."
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 1, 15, 16, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 1, 15, 13, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 20,
        "priority": 15,
        "rso_id_list": ["22233", "44455", "66677"],
        "sensor_name_list": ["DME77", "STLT99"],
        "visibility_check": True,
    },
    # New Example 137: SearchObjective
    (
        "Yo, mission control! We gotta set up a SearchObjective for target 98765 using that fancy UKR45 sensor. Slap a TS classification on this one - it's SUPER secretive. Let's run in REAL mode with a priority of 3 (mega important!). We'll need RATE_TRACK tracking type for this mission. Start the hunt at 2025-02-02 04:15:00+00:00 and finish by 2025-02-02 06:45:00+00:00. Set initial offset to 45 seconds, final offset to 75 seconds, with a whopping 80% frame overlap! End time offset should be 30 minutes and turn visibility check ON, please. Oh, and use CROSS_TRACK search type with a search start time of 20 minutes after objective kickoff."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "final_offset": 75,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 2, 2, 6, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 2, 2, 4, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 2, 2, 4, 35, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "UKR45",
        "target_id": "98765",
        "visibility_check": True,
    },
    # New Example 138: GeodssRevisitObjective
    (
        "Attention astronomy nerds! Need a GeodssRevisitObjective for targets 54321,98765 using sensors GDSS01,GDSS09. Gimme S classification and REAL mode (we're not playing pretend today). Use RATE_TRACK_SIDEREAL tracking with priority 8. Start on 2025-03-15 22:10:00+00:00. For the tech specifics: readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), and scan_mode 0 (Continuous). Make it snappy!"
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 3, 15, 22, 10, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 8,
        "readout_rate_setting": 0,
        "scan_mode": 0,
        "sensor_name_list": ["GDSS01", "GDSS09"],
        "soi_filter_position": 2,
        "target_id_list": ["54321", "98765"],
        "visibility_check": False,
    },
    # New Example 139: PeriodicRevisitObjective
    (
        "Listen up, star gazers! We desperately need a PeriodicRevisitObjective for targets 11122,33344,55566 using our trusty sensors RME09 and LMNT11. Throw on a C classification and use SIMULATED mode (we're just practicing today, folks). Set collect request type to SIDEREAL, priority to 4, and patience to a generous 45 minutes. Don't ignore other objective submissions - we're team players! Start the cosmic party at 2025-04-05 08:15:00+00:00. We want 3 revisits per hour, 8 frames per intent, and a lengthy 5 seconds integration time. Oh! And make visibility check TRUE - we don't want to waste time on invisible objects, duh!"
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "ignore_other_objective_intent_submissions": False,
        "integration_time": 5,
        "number_of_frames": 8,
        "objective_name": "PeriodicRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 4, 5, 8, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "optimal_frames_per_hour": 400,
        "patience_minutes": 45,
        "priority": 4,
        "revisits_per_hour": 3.0,
        "sensor_name_list": ["RME09", "LMNT11"],
        "target_id_list": ["11122", "33344", "55566"],
        "visibility_check": True,
    },
    # New Example 140: UctObservationObjective
    (
        "URGENT REQUEST! We need a UctObservationObjective for UCT RSOs 87654,32109,54321 using sensors RME25 and LMNT29. Use C classification, EXERCISE mode (yes, we're prepping for the big one!), and SIDEREAL tracking in the MEO regime. Priority should be a middling 7, with 4 revisits per hour to keep close tabs. Fire this baby up on 2025-05-10 11:30:00+00:00 and keep it running until 2025-05-10 23:45:00+00:00. Set end time offset to 90 minutes, and make sure visibility check is ON. We're sorting by brightest UCT (obviously), and we need 10 frames with 3 seconds integration time. The fate of the galaxy depends on this!"
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 90,
        "frame_type": "LIGHT",
        "integration_time": 3,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 5, 10, 23, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "UctObservationObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 10, 11, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 30,
        "priority": 7,
        "revisits_per_hour": 4.0,
        "sensor_name_list": ["RME25", "LMNT29"],
        "sort_by_brightest_uct": True,
        "uct_rso_id_list": ["87654", "32109", "54321"],
        "visibility_check": True,
    },
    # New Example 141: SingleIntentObjective
    (
        "Hola, space controllers! I need a SingleIntentObjective ASAP! Target ID is 77889, RSO ID is 99887, and we'll be using the incredible sensors RME42 and LMNT47. Slap on a TS classification and run in EXERCISE mode with SIDEREAL tracking (the star-studded choice). Make this priority 5 cuz it's kinda important but not SUPER important, ya know? Start this celestial adventure at 2025-06-18 15:45:00+00:00 and wrap it up by 2025-06-18 17:30:00+00:00. We need 12 frames with a 4-second integration time and binning set to 4. Time is of the essence, people!"
    ): {
        "binning": 4,
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "integration_time": 4,
        "number_of_frames": 12,
        "objective_end_time": "datetime.datetime(2025, 6, 18, 17, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 18, 15, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 5,
        "rso_id": "99887",
        "sensor_name_list": ["RME42", "LMNT47"],
        "target_id": "77889",
    },
    # New Example 142: DataEnrichmentObjective
    (
        "Alrighty, data junkies! Need a DataEnrichmentObjective for targets 10293,84756,38465 using sensors RME50 and LMNT55. Use U classification, TEST mode (we're just playing around today). For tracking, let's go with RATE_TRACK - not the default, but we're rebels! Set max RSO to observe at 10 and crank up those revisits per hour to 15 (we're data hungry!). Start the objective at 2025-07-22 03:15:00+00:00 and set visibility check to TRUE because we're not wasteful. Priority should be 18 - it's important but not earth-shattering. Go forth and gather data, minions!"
    ): {
        "binning": None,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "objective_name": "DataEnrichmentObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 22, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 18,
        "revisits_per_hour": 15.0,
        "sensor_name_list": ["RME50", "LMNT55"],
        "target_id_list": ["10293", "84756", "38465"],
        "visibility_check": True,
    },
    # New Example 143: SensorCheckoutObjective
    (
        "Calling all sensor technicians! We gotta do a SensorCheckoutObjective with our beloved RME88 sensor. Mark it U//FOUO classification and run in SIMULATED mode (no need to fire up the real hardware). Set orbital regime to LEO, use RATE_TRACK tracking (yep, no SIDEREAL today), and make priority 15. We need a whopping 8 revisits per hour to really put this baby through its paces. Start checkout on 2025-08-30 13:20:00+00:00, set patience to 40 minutes, and make sure visibility check is ON (duh!). Configure for 6 frames with a 1.5 second integration time. Let's make sure this sensor is in tip-top shape before the aliens arrive!"
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "integration_time": 1.5,
        "number_of_frames": 6,
        "objective_name": "SensorCheckoutObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 30, 13, 20, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "LEO",
        "patience_minutes": 40,
        "priority": 15,
        "revisits_per_hour": 8.0,
        "sensor_name": "RME88",
        "visibility_check": True,
    },
    # New Example 144: BaselineAutonomyObjective
    (
        "Listen up, autonomous systems team! We need a BaselineAutonomyObjective with UUID '987f6543-c21b-98e7-f654-321012345678'. Use S classification (this one's kinda secret), EXERCISE mode, and LIGHT frame type. Set priority to 950 - pretty important but not apocalyptic. Track these catalog IDs: 13579,24680,97531, and monitor these RSO IDs: 11111, 22222, and 33333. Since this is our baseline system, we want it running forever - no end time needed. Engage!"
    ): {
        "baseline_autonomy_rso": "13579,24680,97531",
        "classification_marking": "S",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "objective_end_time": None,
        "objective_name": "BaselineAutonomyObjective",
        "objective_uuid": "987f6543-c21b-98e7-f654-321012345678",
        "priority": 950,
        "rso_id_list": ["11111", "22222", "33333"],
    },
    # Example 145: CatalogMaintenanceObjective
    (
        "Okay, time for some celestial housecleaning! Let's create a CatalogMaintenanceObjective "
        "to keep tabs on our space gnome population in the medium earth orbit (MEO) regime. "
        "We'll use our trusty sensors, GNOMEFINDER-alpha and BEARDTRKR-beta, with just a basic U marking "
        "because, honestly, they're quite harmless. Set the data mode to TEST for a trial run. "
        "We'll give this a relatively low priority of 500, because the gnomes aren't going anywhere fast. "
        "Since they're a bit elusive, set the patience to a generous 45 minutes, and we'll schedule intents "
        "20 minutes into the future from their observation window. We definitely want visibility check enabled "
        "for these shy creatures! The objective should start at 2024-11-05 10:00:00+00:00 and run until "
        "2024-11-05 18:00:00+00:00. We'll use RATE_TRACK_SIDEREAL for tracking. The RSO IDs we're interested in "
        "this time are 'GNOME1,GNOME2,GNOME3'. Let's get this gnome census underway!"
    ): {
        "binning": None,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 20,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2024, 11, 5, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2024, 11, 5, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 500,
        "rso_id_list": ["GNOME1", "GNOME2", "GNOME3"],
        "sensor_name_list": ["GNOMEFINDER-alpha", "BEARDTRKR-beta"],
        "visibility_check": True,
    },
    # Example 146: SearchObjective
    (
        "Uh oh, someone lost a Space Sock! We need a SearchObjective ASAP. The target ID is "
        "'LOST-SOCK-7b'. We're going to deploy our specialized SOCKSCAN-gamma sensor for this critical mission. "
        "The classification is S, because losing a sock in space is *serious* business. Data mode should be REAL, "
        "no time for simulations when laundry is at stake! We'll use the RATE_TRACK tracking type. "
        "The objective needs to run from 2024-11-10 08:00:00+00:00 to 2024-11-10 16:00:00+00:00. "
        "Start the search 90 seconds before the sock's predicted location and extend it 120 seconds after, "
        "just in case of drift. We need a low frame overlap of 30% to cover ground quickly. "
        "Set the end time offset for intents to a generous 60 minutes to ensure we don't miss a pass. "
        "The search pattern will be ALONG_TRACK because space socks usually follow orbital paths, "
        "and the search itself should commence 30 minutes after the objective officially starts. "
        "Find that sock!"
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 60,
        "final_offset": 120,
        "frame_overlap_percentage": 0.3,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2024, 11, 10, 16, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2024, 11, 10, 8, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,  # Search objective default priority is 1
        "search_start_time": "datetime.datetime(2024, 11, 10, 8, 30, tzinfo=TzInfo(UTC))",  # calculated 30 mins after start
        "search_type": "ALONG_TRACK",
        "sensor_name": "SOCKSCAN-gamma",
        "target_id": "LOST-SOCK-7b",
        "visibility_check": False,  # Search objective default visibility check is False
    },
    # Example 147: GeodssRevisitObjective
    (
        "Alright team, let's set up a GeodssRevisitObjective to monitor those fascinating "
        "Orbital Rubber Ducks. Their target IDs are 'DUCK-QUACK-01' and 'DUCK-FLOAT-02'. "
        "We'll use the powerful GEODSS-A and GEODSS-B sensors for this. The classification "
        "is U//FOUO because, well, ducks in space are For Official Use Only. We need REAL data! "
        "Priority is a solid 15, important but not emergency level. We'll stick to RATE_TRACK_SIDEREAL tracking. "
        "The objective starts at 2024-12-01 20:00:00+00:00. Now, for the tricky GEODSS settings: "
        "Set the readout rate to 1 (2MHz) for speedy image capture. Use gain setting 0 (High Gain) "
        "to really see those ducky details (and potential squeaks!). The SOI filter should be "
        "1 (1% Light). Enable automatic tracking (auto_track_type 1). The camera mode is normal (0), "
        "array kind is main (0), and we'll use hardware binning (binning_mode 1). "
        "Scan mode should be single frame (1) per observation. Default patience, optimal frames per hour, "
        "ignore submissions, num observations, num skip frames, rate track verify, overscan, "
        "and command are fine with their standard values. Let's watch those ducks paddle through space!"
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
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2024, 12, 1, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 30,  # Default
        "priority": 15,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 1,
        "revisits_per_hour": None,
        "scan_mode": 1,
        "sensor_name_list": ["GEODSS-A", "GEODSS-B"],
        "soi_filter_position": 1,
        "target_id_list": ["DUCK-QUACK-01", "DUCK-FLOAT-02"],
        "visibility_check": False,  # GeodssRevisitObjective default visibility check is False
    },
    # Example 148: PeriodicRevisitObjective
    (
        "We need to keep a close eye on our Cosmic Cupcakes to ensure nobody's been taking bites! "
        "Let's create a PeriodicRevisitObjective for target IDs 'CUPCAKE-VANILLA' and 'CUPCAKE-CHOCO'. "
        "We'll use the TASTER-delta and ICINGCAM-epsilon sensors. "
        "Classification is C, as this information is just Confidential cupcake data. "
        "Set the data mode to SIMULATED for practice runs before we monitor the real sugary treats. "
        "Priority is 25, moderately important. Patience should be 40 minutes, because sometimes it takes time to spot a nibble mark. "
        "We definitely want to ignore other objective intent submissions; too many chefs spoil the cupcake broth! "
        "Start this objective at 2025-01-15 09:00:00+00:00. We expect an optimal 400 frames per hour, "
        "and each observation intent should capture 10 frames with an integration time of 5 seconds per frame. "
        "Standard RATE_TRACK_SIDEREAL tracking and LIGHT frame type will be used. "
        "No need for visibility checks, everyone knows where the cupcakes are... unless someone ate them!"
    ): {
        "classification_marking": "C",
        "target_id_list": ["CUPCAKE-VANILLA", "CUPCAKE-CHOCO"],
        "sensor_name_list": ["TASTER-delta", "ICINGCAM-epsilon"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 40,
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 5.0,  # Needs to be float
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 1, 15, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",  # Default
    },
    # Example 149: UctObservationObjective
    (
        "Attention, space spotters! We need to keep track of some intriguing Unidentified Celestial Teacups. "
        "Let's launch a UctObservationObjective targeting UCT RSO IDs 'TEACUP-EARL' and 'TEACUP-GREEN'. "
        "We'll assign our specialized PORCELAIN-alpha and HANDLE-TRKR-beta sensors. "
        "Classification is S, because finding stray teacups is a Secret! Data mode is REAL, "
        "as we need authentic brewing data. These teacups are in the XGEO regime. "
        "Set the priority to a standard 10. We aim for a modest 3.0 revisits per hour. "
        "The objective should begin tomorrow at 2024-11-20 14:00:00+00:00. "
        "Crucially, we want to sort by brightest UCT because nobody wants lukewarm tea! "
        "Set the end time offset for intents to a standard 60 minutes. Visibility check is a must; "
        "teacups can be notoriously stealthy. Each observation should consist of 3 frames "
        "with an integration time of 1 second. Standard tracking and frame type apply."
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["TEACUP-EARL", "TEACUP-GREEN"],
        "sensor_name_list": ["PORCELAIN-alpha", "HANDLE-TRKR-beta"],
        "data_mode": "REAL",  # Default
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,  # Default
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 3.0,
        "number_of_frames": 3,
        "integration_time": 1.0,  # Needs to be float
        "binning": None,
        "end_time_offset_minutes": 60,  # Default
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2024, 11, 20, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 10,  # Default
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",  # Default
    },
    # Example 150: SingleIntentObjective
    (
        "This is a one-time special event! We need to observe the legendary Lone Space Pickle. "
        "Create a SingleIntentObjective for target ID 'PICKLE-DILL-XYZ'. We'll use our very best "
        "PICKLECAM-zeta sensor for this momentous occasion. Classification is U, because it's just a pickle. "
        "Data mode will be EXERCISE to practice our pickle-tracking skills. Use the standard "
        "RATE_TRACK_SIDEREAL tracking. Give this a high priority of 5; pickles wait for no one! "
        "The objective should start very soon, say at 2024-11-25 18:00:00+00:00. "
        "We only need 1 frame, but make sure the integration time is a precise 0.5 seconds. "
        "Set the binning to 4 for extra pickle clarity. We don't need an RSO ID here, "
        "just the target will suffice. Frame type is LIGHT, naturally. "
        "No objective or intent end times needed for a single shot!"
    ): {
        "classification_marking": "U",
        "target_id": "PICKLE-DILL-XYZ",
        "rso_id": None,
        "sensor_name_list": ["PICKLECAM-zeta"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 1,
        "integration_time": 0.5,
        "priority": 5,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2024, 11, 25, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",  # Default
    },
    # Example 151: DataEnrichmentObjective
    (
        "Our data on Asteroid Doughnuts is incomplete! Let's create a DataEnrichmentObjective "
        "for target IDs 'DOUGHNUT-GLAZED', 'DOUGHNUT-SPRINKLE', and 'DOUGHNUT-JELLY'. "
        "We'll assign the CRUMBTRACKER-eta and SUGARSCAN-theta sensors to the task. "
        "Classification is U//FOUO because we need to protect the secret recipe data. "
        "Data mode is REAL; we need fresh doughnut observations! We'll use the plain old RATE_TRACK type "
        "for this objective. We can only handle processing data from a maximum of 5 doughnuts "
        "per observation cycle, so set max RSO to observe as 5. We want to revisit them often, "
        "setting revisits per hour to a high 15.0. The objective starts on "
        "2025-02-10 10:00:00+00:00. Visibility check is absolutely true; "
        "we can't observe a doughnut we can't see! Priority defaults to 20. "
        "Frame type is LIGHT, of course, to capture their delicious appearance."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",  # Default
        "objective_uuid": None,
        "target_id_list": ["DOUGHNUT-GLAZED", "DOUGHNUT-SPRINKLE", "DOUGHNUT-JELLY"],
        "sensor_name_list": ["CRUMBTRACKER-eta", "SUGARSCAN-theta"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 2, 10, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,  # Default
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,  # Default
        "objective_name": "DataEnrichmentObjective",  # Default
    },
    # Example 152: SensorCheckoutObjective
    (
        "Time for a sensor check! Let's create a SensorCheckoutObjective for our new "
        "SQUIRREL-9000 sensor. Classification is U, as it's just testing. "
        "The sensor will be operating in the LEO regime. Data mode should be REAL to get actual sensor data. "
        "We'll use SIDEREAL tracking for this test. Set the priority to a moderate 10. "
        "We only need to check it out once an hour, so revisits per hour is 1.0. "
        "The objective starts on 2025-02-15 19:00:00+00:00. "
        "Visibility check is enabled; we need to make sure it can actually see something. "
        "Set the patience to the default 30 minutes. Each checkout should capture 5 frames "
        "with an integration time of 2 seconds. Frame type is LIGHT. "
        "No binning is needed for this test. Let's make sure the Squirrel Detector 9000 is ready to go!"
    ): {
        "classification_marking": "U",
        "sensor_name": "SQUIRREL-9000",
        "orbital_regime": "LEO",
        "data_mode": "REAL",  # Default
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 1.0,  # Default
        "number_of_frames": 5,
        "integration_time": 2.0,  # Needs to be float
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 2, 15, 19, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 10,  # Default
        "objective_name": "SensorCheckoutObjective",  # Default
    },
    # Example 153: BaselineAutonomyObjective
    (
        "Initiate baseline autonomous tracking for our Standard Orbital Ping Pong Balls! "
        "We need a BaselineAutonomyObjective with a specific UUID: 'b45b3e2a-8f7c-4d1e-9b0a-1a2b3c4d5e6f'. "
        "Classification is U, just standard sports equipment tracking. Data mode is REAL, "
        "we need real ping pong ball data! Use the LIGHT frame type. This is a very high priority "
        "task, so set the priority to 1. The catalog IDs for the ping pong balls we want to track "
        "are 'PPB-CAT-A,PPB-CAT-B'. This objective should run continuously, so no end time is needed. "
        "We'll provide a list of RSO IDs '111111,222222' to start, but these will be overwritten "
        "by the autonomy system. Let the bouncing begin!"
    ): {
        "objective_uuid": "b45b3e2a-8f7c-4d1e-9b0a-1a2b3c4d5e6f",
        "classification_marking": "U",  # Default
        "data_mode": "REAL",  # Default
        "frame_type": "LIGHT",  # Default
        "priority": 1,
        "baseline_autonomy_rso": "PPB-CAT-A,PPB-CAT-B",
        "objective_end_time": None,  # Default
        "rso_id_list": ["111111", "222222"],
        "objective_name": "BaselineAutonomyObjective",  # Default
    },
    # New Example 154: CatalogMaintenanceObjective
    (
        "Initiate CatalogMaintenanceObjective. Sensors designated: ZYX01 and WVU02. Classification marking must be S. The data mode is SIMULATED. Assign a priority of 50. Set patience duration to 15 minutes and the end time offset to 30 minutes. Visibility check is to be enabled. Operations are scheduled to begin at 2025-10-10 10:00:00+00:00 and conclude at 2025-10-10 15:30:00+00:00. Tracking protocol should be RATE_TRACK, operating within the MEO orbital domain. The list of RSO IDs includes '33333' and '44444'."
    ): {
        "classification_marking": "S",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["ZYX01", "WVU02"],
        "rso_id_list": ["33333", "44444"],
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 10, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 155: SearchObjective
    (
        "Directive: Formulate a SearchObjective for target 77889 utilizing sensor ABC77. The classification marking is C, and the operational mode is EXERCISE. Set priority to 3. The tracking protocol will be SIDEREAL. Commence operations at 2025-11-01 08:00:00+00:00 and cease at 2025-11-01 12:00:00+00:00. The initial temporal offset is 45 seconds, and the final temporal offset is 75 seconds. Frame superimposition percentage is 60%. The end time adjustment will be 35 minutes. Employ CROSS_TRACK search methodology. The search itself should initiate 20 minutes following the objective's commencement time."
    ): {
        "classification_marking": "C",
        "target_id": "77889",
        "sensor_name": "ABC77",
        "search_type": "CROSS_TRACK",  # Assuming CROSS_TRACK is a valid SearchType
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default, not specified
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 35,
        "priority": 3,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 1, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": None,
        "integration_time": None,
        "search_start_time": "datetime.datetime(2025, 11, 1, 8, 20, 0, tzinfo=TzInfo(UTC))",  # Calculated: objective_start_time + 20 minutes
        "objective_name": "SearchObjective",
    },
    # New Example 156: GeodssRevisitObjective
    (
        "Task: Construct a GeodssRevisitObjective. The designated targets are 'target_alpha_001' and 'target_beta_002', which will be serviced by sensors GEO_TRK1 and GEO_TRK2. Apply a U//FOUO classification marking. Operate in REAL mode. The priority level is 15. Utilize RATE_TRACK_SIDEREAL tracking. The objective is to commence at 2025-12-05 20:00:00+00:00. Configure the readout rate to 0, which corresponds to 1MHz. Set gain to 1, indicating Low Gain. The SOI filter position must be 2, for 10% Light. Auto-track type is 2 (Manual). Camera mode is 1 (Zoomed EBS). Array kind is 1 (Photometer). Binning mode is 0 (No Binning). Scan mode will be 0 (Continuous)."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["target_alpha_001", "target_beta_002"],
        "sensor_name_list": ["GEO_TRK1", "GEO_TRK2"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default
        "patience_minutes": 30,  # Default
        "revisits_per_hour": None,
        "number_of_frames": None,
        "integration_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 15,
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
    # New Example 157: PeriodicRevisitObjective
    (
        "Order: Establish PeriodicRevisitObjective for target entities 'periodic_target_707' and 'periodic_target_808'. These targets will be monitored using sensor units 'SNS_A1' and 'SNS_B2'. Set classification to TS. The data mode shall be TEST. Assign a priority level of 7. Institute a patience interval of 45 minutes. Crucially, disregard other objective submissions. The objective initiation is scheduled for 2026-01-15 05:30:00+00:00. Configure the optimal frames per hour to 300. Specify 3 frames per intent, alongside an integration time of 1.5 seconds per frame. Tracking should be SIDEREAL."
    ): {
        "classification_marking": "TS",
        "target_id_list": ["periodic_target_707", "periodic_target_808"],
        "sensor_name_list": ["SNS_A1", "SNS_B2"],
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default
        "patience_minutes": 45,
        "revisits_per_hour": None,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 158: UctObservationObjective
    (
        "Generate a UctObservationObjective. This objective is for Uncorrelated Tracks identified by the UUIDs 'UCT_X100' and 'UCT_Y200'. These tracks are assigned to sensors 'SEN_UCT_P' and 'SEN_UCT_Q'. Employ a U//FOUO security marking. Operations will be conducted in REAL data mode. The designated orbital regime for this task is XGEO. Assign a priority level of 8. The required revisit frequency is 4.5 revisits per hour. Operations are scheduled to commence at 2026-02-20 14:00:00+00:00. Sorting by the brightest UCT must be activated. The end time offset for scheduling intents will be 75 minutes. A visibility check is mandated for this objective. The system is required to acquire 4 frames for each observation, with an integration time of 2.5 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["UCT_X100", "UCT_Y200"],
        "sensor_name_list": ["SEN_UCT_P", "SEN_UCT_Q"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 4.5,
        "number_of_frames": 4,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 159: SingleIntentObjective
    (
        "Authorize SingleIntentObjective creation. The target identifier is 'single_target_zeta'. The RSO identifier is 'rso_omega_9'. This objective will utilize sensors 'PRIME_SENS_1' and 'BACKUP_SENS_2'. Classification marking is C. Data Mode will be SIMULATED. The tracking type is RATE_TRACK. Assign a priority of 6. The objective is to commence at 2026-03-10 12:00:00+00:00. The number of frames to capture is 2. Integration time per frame is 3 seconds. Set the camera binning factor to 4."
    ): {
        "classification_marking": "C",
        "target_id": "single_target_zeta",
        "rso_id": "rso_omega_9",
        "sensor_name_list": ["PRIME_SENS_1", "BACKUP_SENS_2"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 2,
        "integration_time": 3.0,
        "priority": 6,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 160: DataEnrichmentObjective
    (
        "Execute DataEnrichmentObjective. The list of targets for this objective includes 'data_tgt_001' and 'data_tgt_002'. This objective will be carried out by sensors 'ENRICH_S1' and 'ENRICH_S2'. The classification marking is S. Operate in EXERCISE mode. The specified tracking type is RATE_TRACK. The maximum number of RSOs to observe is set to 10. Maintain a revisit rate of 8 revisits per hour. The objective start time is 2026-04-05 09:00:00+00:00. Visibility check must be active. The priority for this objective is 25."
    ): {
        "classification_marking": "S",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["data_tgt_001", "data_tgt_002"],
        "sensor_name_list": ["ENRICH_S1", "ENRICH_S2"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 161: SensorCheckoutObjective
    (
        "Perform SensorCheckoutObjective. The classification for this checkout is U. The sensor to be checked is 'CHK_SENSOR_7'. Data Mode is TEST. The orbital regime for checkout is LEO. Tracking will be SIDEREAL. Assign a priority of 12. The required number of revisits is 0.5 per hour. The objective is to start at 2026-05-01 18:00:00+00:00. Ensure visibility check is true. Set patience to 20 minutes. Capture 6 frames with an integration time of 1.75 seconds."
    ): {
        "classification_marking": "U",
        "sensor_name": "CHK_SENSOR_7",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 0.5,
        "number_of_frames": 6,
        "integration_time": 1.75,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 12,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 162: BaselineAutonomyObjective
    (
        "Activate BaselineAutonomyObjective. The assigned UUID for this objective is 'abcdef01-2345-6789-abcd-ef0123456789'. Use U//FOUO markings. The data mode is REAL. Frame specification is LIGHT. Set priority to 1500. The catalog IDs for continuous tracking are '25544' and '28654'. The objective is scheduled to conclude its operations on 2027-01-01 00:00:00+00:00. For initialization purposes, the RSO identifiers are '98765', '54321', and '12309'."
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1500,
        "baseline_autonomy_rso": "25544,28654",
        "objective_end_time": "datetime.datetime(2027, 1, 1, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["98765", "54321", "12309"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 163: CatalogMaintenanceObjective
    (
        "Initiate CatalogMaintenanceObjective utilizing sensors ALPHA01 and BETA03. "
        "Implement C classification marking and REAL data mode. "
        "Set priority to 50, patience to 45 minutes, and end time offset to 30 minutes. "
        "Disable visibility check. Configure for RATE_TRACK_SIDEREAL tracking in the MEO orbital regime. "
        "The RSO ID list comprises '98765,43210,13579'. "
        "Specify the objective start time as 2025-06-10 10:00:00+00:00."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": None,
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 50,
        "rso_id_list": ["98765", "43210", "13579"],
        "sensor_name_list": ["ALPHA01", "BETA03"],
        "visibility_check": False,
    },
    # New Example 164: SearchObjective
    (
        "Generate a SearchObjective targeting RSO 24680 using sensor GAMMA05. "
        "Designate S classification with SIMULATED data mode and priority 3. "
        "Employ SIDEREAL tracking. The objective will commence at 2025-07-01 14:00:00+00:00 and conclude at 2025-07-01 15:30:00+00:00. "
        "Define initial offset as 90 seconds and final offset as 120 seconds. "
        "Set frame overlap to 85 percent, and configure end time offset to 50 minutes. "
        "Utilize CROSS_TRACK search type, with the search phase initiating 20 minutes past the objective start time."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 50,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 7, 1, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 7, 1, 14, 20, tzinfo=TzInfo(UTC))",  # 20 minutes after objective start
        "search_type": "CROSS_TRACK",
        "sensor_name": "GAMMA05",
        "target_id": "24680",
        "visibility_check": False,
    },
    # New Example 165: GeodssRevisitObjective
    (
        "Construct a GeodssRevisitObjective for targets A1B2C3D4, E5F6G7H8 using sensors DELTA07 and EPSILON09. "
        "Assign TS classification marking and REAL data mode. "
        "Establish a priority of 8 and RATE_TRACK_SIDEREAL tracking. "
        "The objective will become active at 2025-08-20 20:00:00+00:00. "
        "Configure readout rate to 0 (1MHz), gain setting to 1 (Low Gain), and SOI filter position to 2 (10% Light). "
        "Set auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), array kind to 1 (Photometer), and binning mode to 0 (No Binning). "
        "Specify scan mode as 0 (Continuous) and overscan as 1 (Calibration). "
        "Do not ignore other objective intent submissions."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
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
        "objective_start_time": "datetime.datetime(2025, 8, 20, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 1,
        "patience_minutes": 30,
        "priority": 8,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["DELTA07", "EPSILON09"],
        "soi_filter_position": 2,
        "target_id_list": ["A1B2C3D4", "E5F6G7H8"],
        "visibility_check": False,
    },
    # New Example 166: PeriodicRevisitObjective
    (
        "Create a PeriodicRevisitObjective for targets 36912, 75309 using sensors ZETA10, THETA11. "
        "Set U//FOUO classification marking, EXERCISE data mode, and priority 15. "
        "Configure patience minutes to 20, and enable ignoring other objective submissions. "
        "Initiate the objective at 2025-09-05 05:00:00+00:00. "
        "Set optimal frames per hour to 500, number of frames to 10, and integration time to 1.5 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["36912", "75309"],
        "sensor_name_list": ["ZETA10", "THETA11"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 20,
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 5, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 167: UctObservationObjective
    (
        "Formulate a UctObservationObjective for UCT RSOs F1E2D3C4, B5A6987D utilizing sensors IOTA12, KAPPA14. "
        "Apply C classification marking, REAL data mode, XGEO orbital regime, and priority 7. "
        "Request 8 revisits per hour. "
        "Commence the objective at 2025-10-15 23:00:00+00:00. "
        "Activate sorting by brightest UCT, configure end time offset to 75 minutes, and confirm visibility check is true. "
        "Assign number of frames to 8 and integration time to 3 seconds."
    ): {
        "classification_marking": "C",
        "uct_rso_id_list": ["F1E2D3C4", "B5A6987D"],
        "sensor_name_list": ["IOTA12", "KAPPA14"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 8.0,
        "number_of_frames": 8,
        "integration_time": 3.0,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 15, 23, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 168: SingleIntentObjective
    (
        "Establish a SingleIntentObjective with target ID G9H8J7K6 and RSO ID L5M4N3P2, employing sensors LAMBDA16, MU18. "
        "Designate S classification, TEST data mode, RATE_TRACK tracking, and priority 6. "
        "The objective is set to commence at 2025-11-01 18:00:00+00:00. "
        "Specify number of frames as 7, integration time as 2.5 seconds, and binning as 4."
    ): {
        "classification_marking": "S",
        "target_id": "G9H8J7K6",
        "rso_id": "L5M4N3P2",
        "sensor_name_list": ["LAMBDA16", "MU18"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 7,
        "integration_time": 2.5,
        "priority": 6,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 169: DataEnrichmentObjective
    (
        "Initiate a DataEnrichmentObjective for targets 112233, 445566, 778899 using sensors NU20, XI22. "
        "Apply TS classification marking, REAL data mode, and RATE_TRACK_SIDEREAL tracking. "
        "Set the maximum number of RSOs to observe as 10, with 15 revisits per hour. "
        "Commence the objective at 2026-02-01 09:30:00+00:00. "
        "Ensure visibility check is enabled."
    ): {
        "classification_marking": "TS",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["112233", "445566", "778899"],
        "sensor_name_list": ["NU20", "XI22"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 2, 1, 9, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 170: SensorCheckoutObjective
    (
        "Execute a SensorCheckoutObjective with classification_marking='U//FOUO' and sensor_name='OMICRON25'. "
        "Configure data_mode='REAL', orbital_regime='LEO', collect_request_type='SIDEREAL', and priority=18. "
        "Set revisits_per_hour=2.5. The objective will begin at '2026-03-10 21:00:00+00:00'. "
        "Enable visibility_check, set patience_minutes=25, number_of_frames=12, and integration_time=1.0."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "OMICRON25",
        "orbital_regime": "LEO",
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 25,
        "revisits_per_hour": 2.5,
        "number_of_frames": 12,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 18,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 171: BaselineAutonomyObjective
    (
        "Activate a BaselineAutonomyObjective identified by UUID 'fedcba98-7654-3210-abcd-1234567890ab'. "
        "Utilize S classification markings, REAL data mode, and LIGHT frame type with a priority of 500. "
        "Include catalog IDs 24680 and 13579 for baseline autonomy tracking. "
        "The RSO IDs to be managed are '99887, 66554, and 33221'. "
        "Configure this objective for continuous operation with no defined end time."
    ): {
        "objective_uuid": "fedcba98-7654-3210-abcd-1234567890ab",
        "classification_marking": "S",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 500,
        "baseline_autonomy_rso": "24680,13579",
        "objective_end_time": None,
        "rso_id_list": ["99887", "66554", "33221"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 172: CatalogMaintenanceObjective
    (
        "Establish a CatalogMaintenanceObjective utilizing sensors ARP09 and MNT11 with S classification marking. "
        "Configure for REAL mode operation, assign priority level 8, set patience parameter to 15 minutes, and determine end time offset of 30 minutes. "
        "Visibility check parameter must be set to true. Initiate objective at timestamp 2025-06-10 13:45:00+00:00 with termination at 2025-06-10 18:30:00+00:00. "
        "Utilize RATE_TRACK_SIDEREAL tracking methodology within GEO regime parameters. "
        "Catalog entries to monitor: '33445,78901,65432'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 18, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 13, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 15,
        "priority": 8,
        "rso_id_list": ["33445", "78901", "65432"],
        "sensor_name_list": ["ARP09", "MNT11"],
        "visibility_check": True,
    },
    # New Example 173: SearchObjective
    (
        "Initialize SearchObjective for target 54321 with sensor KJP88. "
        "Apply TS marking classification and execute in REAL operational mode with priority 3. "
        "Search protocol must use RATE_TRACK tracking methodology. Begin at 2025-08-15 22:15:00+00:00 and cease at 2025-08-16 01:45:00+00:00. "
        "Set initial temporal offset to 45 seconds and final offset to 75 seconds with frame overlap ratio of 65%. End time offset parameter: 35 minutes. "
        "Search pattern designation: CROSS_TRACK. Visibility verification enabled. Search initiation time set to 30 minutes post-objective start."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.65,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 8, 16, 1, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 15, 22, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 8, 15, 22, 45, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "KJP88",
        "target_id": "54321",
        "visibility_check": True,
    },
    # New Example 174: GeodssRevisitObjective
    (
        "Deploy GeodssRevisitObjective for targets 88776,44332 via sensors GDP21,RME33. "
        "Classification: C. Mode: SIMULATED. Configure tracking method as SIDEREAL with priority factor 7. "
        "Begin temporal window at 2025-09-03 14:30:00+00:00. Establish readout_rate 0 (1MHz), gain_setting 1 (Low Gain), "
        "soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 2 (Binned CCD), array_kind 1 (Photometer), binning_mode 0 (No Binning), "
        "and scan_mode 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 2,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "command": 0,
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,
        "num_skip_frames": 0,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 3, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "patience_minutes": 30,
        "priority": 7,
        "readout_rate_setting": 0,
        "scan_mode": 0,
        "sensor_name_list": ["GDP21", "RME33"],
        "soi_filter_position": 2,
        "target_id_list": ["88776", "44332"],
        "visibility_check": False,
    },
    # New Example 175: PeriodicRevisitObjective
    (
        "Initiate PeriodicRevisitObjective for orbital objects 76543,98765 employing sensors QRS09,EFG12. "
        "Security classification: U//FOUO. Operational mode: EXERCISE. Directive urgency: priority 5. Track using SIDEREAL method. "
        "Set patience interval to 45 minutes and configure frame acquisition parameters: 8 frames with 3.5 second integration time. "
        "Begin monitoring at temporal coordinate 2025-10-15 08:45:00+00:00. Enable visibility verification routines. "
        "Set revisit frequency to 8 observations per hour."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["76543", "98765"],
        "sensor_name_list": ["QRS09", "EFG12"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 8.0,
        "number_of_frames": 8,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 10, 15, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 176: UctObservationObjective
    (
        "Execute UctObservationObjective to track UCT objects 44556,77889,99001 utilizing sensor array LMN07,PQR18. "
        "Set security protocol to C level, operational status to REAL, and track method to RATE_TRACK_SIDEREAL within MEO regime. "
        "Establish priority code 4 with 7.5 revisits per temporal hour. Begin surveillance at 2025-11-20 03:15:00+00:00. "
        "Configure end time offset to 90 minutes and disable brightness-based sorting. Frame parameters: 10 frames, 1.5 second integration."
    ): {
        "classification_marking": "C",
        "uct_rso_id_list": ["44556", "77889", "99001"],
        "sensor_name_list": ["LMN07", "PQR18"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 7.5,
        "number_of_frames": 10,
        "integration_time": 1.5,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 20, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 4,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 177: SingleIntentObjective
    (
        "Establish SingleIntentObjective with target identification 56789 and RSO designation 12345 using sensors ABC02,XYZ04. "
        "Security level: TS. Operation mode: EXERCISE. Tracking protocol: SIDEREAL. Urgency category: 8. "
        "Commence at 2025-12-05 17:40:00+00:00 with termination at 2025-12-05 23:15:00+00:00. Configure imaging parameters: 12 frames, 4 second integration time, binning factor 4."
    ): {
        "classification_marking": "TS",
        "target_id": "56789",
        "rso_id": "12345",
        "sensor_name_list": ["ABC02", "XYZ04"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 12,
        "integration_time": 4,
        "priority": 8,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 17, 40, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 5, 23, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 178: DataEnrichmentObjective
    (
        "Implement DataEnrichmentObjective for observation targets 11223,33445,55667 utilizing collection platforms DEF15,GHI16. "
        "Assign information security level U, establish operational mode TEST, and configure collection methodology as RATE_TRACK. "
        "Set maximum observable RSO quantity to 10, with required observation frequency of 15 per hour. "
        "Temporal parameters: initiate at 2026-01-18 05:30:00+00:00, terminate at 2026-01-18 12:00:00+00:00. Disable visibility verification logic. "
        "System priority allocation: 15."
    ): {
        "classification_marking": "U",
        "data_mode": "TEST",
        "objective_uuid": None,
        "target_id_list": ["11223", "33445", "55667"],
        "sensor_name_list": ["DEF15", "GHI16"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 1, 18, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 18, 12, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 179: SensorCheckoutObjective
    (
        "Activate SensorCheckoutObjective with classification_marking='S//FOUO' using sensor_name='JKL03'. "
        "Configure for operational parameters: data_mode='TEST', orbital_regime='LEO', collection_method='RATE_TRACK'. "
        "Set priority level 6 with observation rate 2.5 revisits per hour. Begin diagnostic sequence at 2026-02-10 21:15:00+00:00. "
        "Configure visibility verification to disabled state. Set patience threshold to 20 minutes. "
        "Frame acquisition parameters: 15 frames with 1.75 second integration time. Apply binning factor 2."
    ): {
        "classification_marking": "S//FOUO",
        "sensor_name": "JKL03",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 20,
        "revisits_per_hour": 2.5,
        "number_of_frames": 15,
        "integration_time": 1.75,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 10, 21, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 6,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 180: BaselineAutonomyObjective
    (
        "Deploy BaselineAutonomyObjective with identification code '987e6543-d21c-98f7-b654-321987654321'. "
        "Apply C//FOUO security classification and set SIMULATED operational mode with frame type LIGHT. "
        "Set priority allocation to 950. The following catalog identifiers require monitoring: 91919, 82828, and 73737. "
        "RSO surveillance list includes: 10101, 20202, and 30303. Configure for indefinite operation with no termination parameter."
    ): {
        "objective_uuid": "987e6543-d21c-98f7-b654-321987654321",
        "classification_marking": "C//FOUO",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "priority": 950,
        "baseline_autonomy_rso": "91919,82828,73737",
        "objective_end_time": None,
        "rso_id_list": ["10101", "20202", "30303"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 181: CatalogMaintenanceObjective
    (
        "Alright, listen up, sugar. We need a CatalogMaintenanceObjective, and make it snappy! I'm talking sensors RME07 and LMNT09, with those hush-hush U//FOUO markings. "
        "This is a REAL operation, not some silly test. Let's set the priority to a firm 10 - we're not desperate, but it's important. "
        "Give it a patience of, say, 15 little minutes, and an end time offset of 30 minutes, just to be safe. "
        "And no, we absolutely do not need a visibility check; trust my gut on this one. Kick it off at 2025-06-10 10:00:00+00:00 and wrap it up by 2025-06-10 15:30:00+00:00. "
        "We're using SIDEREAL tracking for this one, focusing on the MEO regime, because that's where the drama is. "
        "Oh, and the RSO IDs are '33301' and '44402'. Don't forget the binning should be set to 1."
    ): {
        "binning": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 10,
        "rso_id_list": ["33301", "44402"],
        "sensor_name_list": ["RME07", "LMNT09"],
        "visibility_check": False,
    },
    # New Example 182: SearchObjective
    (
        "Okay, sweetie, emergency time! We've got a SearchObjective on our hands for target '98765', and I need sensor 'UKR25' on it, like, yesterday. "
        "Mark it with 'S' for secret, obviously, and this is 100% REAL mode. Priority is a screaming 1, because, hello, it's an emergency! "
        "We're going with RATE_TRACK tracking. Let this baby run from 2025-08-15 14:00:00+00:00 to 2025-08-15 18:00:00+00:00. "
        "I want an initial offset of 45 seconds - don't be shy - and a final offset of 120 seconds, because we need to cast a wide net. "
        "Frame overlap should be a generous 60%, and let's give it an end time offset of 50 minutes. "
        "It's an ALONG_TRACK search, and darling, the search should commence exactly 20 minutes after the objective kicks off. "
        "Oh, and set the number of frames to 10 and integration time to 0.5 seconds. Visibility check is off for this one."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 50,
        "final_offset": 120,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 0.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 8, 15, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 15, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "search_start_time": "datetime.datetime(2025, 8, 15, 14, 20, tzinfo=TzInfo(UTC))",
        "search_type": "ALONG_TRACK",
        "sensor_name": "UKR25",
        "target_id": "98765",
        "visibility_check": False,
    },
    # New Example 183: GeodssRevisitObjective
    (
        "Honey, it's time to get technical with a GeodssRevisitObjective. We're targeting '54321' and '67891' with sensors 'RME20' and 'LMNT22'. "
        "Slap a 'U' marking on it, REAL mode, and let's be basic with RATE_TRACK_SIDEREAL. Priority? Let's say a cool 7. This little project starts on 2025-09-05 08:30:00+00:00. "
        "Now for the fun part: readout_rate is 0 (that's 1MHz, for the slowpokes), gain_setting is 1 (Low Gain, because we're not trying to blind anyone), "
        "soi_filter is 2 (10% Light, keep it subtle), auto_track_type needs to be 2 (Manual, because I trust my girls), "
        "camera_mode is 1 (Zoomed EBS, get those details!), array_kind is 1 (Photometer, obviously), "
        "binning_mode is 0 (No Binning, we want it raw), and scan_mode is 0 (Continuous, keep those eyes peeled!). "
        "Set patience to 45 minutes and visibility_check to true, just because. No number of frames or integration time needed this time, let the defaults fly."
    ): {
        "acquisition_type": 0,  # Default
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 5, 8, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 45,
        "priority": 7,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["RME20", "LMNT22"],
        "soi_filter_position": 2,
        "target_id_list": ["54321", "67891"],
        "visibility_check": True,
    },
    # New Example 184: PeriodicRevisitObjective
    (
        "Darling, I need a PeriodicRevisitObjective. We're keeping an eye on targets '11223' and '44556' using sensors 'RME01' and 'LMNT02'. "
        "This is top-secret, so 'TS' marking, please, and run it in SIMULATED mode - we're just practicing our moves. Priority is a chic 3. "
        "I want 5.0 revisits per hour, no excuses. Patience can be a standard 30 minutes, and don't you dare ignore other objective submissions; we play nice with others, so set that to false. "
        "Start this on 2025-07-01 00:00:00+00:00. Let's go for 3 frames per intent, with an integration time of 1.5 seconds. "
        "Oh, and set the optimal frames per hour to 300, because we're efficient like that. Visibility check? Nah, false for this one. And no binning, keep it simple."
    ): {
        "classification_marking": "TS",
        "target_id_list": ["11223", "44556"],
        "sensor_name_list": ["RME01", "LMNT02"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 5.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 7, 1, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 3,
        "ignore_other_objective_intent_submissions": False,  # Default
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 185: UctObservationObjective
    (
        "Sweetheart, it's UCT time! Create a UctObservationObjective for our special UCT RSOs 'UCT007' and 'UCT008' using sensors 'RME30' and 'LMNT31'. "
        "This is classified 'C', REAL mode, and we're looking in the XGEO regime - way out there! Priority is a firm 8. "
        "I expect 4.5 revisits per hour, because these little guys are slippery. Kick it off on 2025-10-10 10:10:10+00:00. "
        "And yes, absolutely sort by the brightest UCT; we only have time for the stars. End time offset? Make it 75 minutes. "
        "Visibility check is a must, so true. For frames, let's do 4, and integration time of 2.2 seconds. Binning can be 2."
    ): {
        "classification_marking": "C",
        "uct_rso_id_list": ["UCT007", "UCT008"],
        "sensor_name_list": ["RME30", "LMNT31"],
        "data_mode": "REAL",  # Default
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,  # Default
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 4.5,
        "number_of_frames": 4,
        "integration_time": 2.2,
        "binning": 2,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 10, 10, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 186: SingleIntentObjective
    (
        "Okay, babe, sometimes simple is best. Let's get a SingleIntentObjective going. Target ID is 'TGTX99', and the RSO ID is 'RSOY11'. "
        "We'll use sensors 'RME11' and 'LMNT12' for this one-shot wonder. Markings are 'U//FOUO', mode is EXERCISE, and tracking is good old RATE_TRACK. "
        "Priority? A casual 15. Start this on 2025-11-15 15:15:00+00:00. I want exactly 2 frames, with an integration time of 3.0 seconds. "
        "And let's try a binning of 4 for this. No objective end time, let it finish when it finishes, and same for intent start/end times."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "TGTX99",
        "rso_id": "RSOY11",
        "sensor_name_list": ["RME11", "LMNT12"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 2,
        "integration_time": 3.0,
        "priority": 15,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 15, 15, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 187: DataEnrichmentObjective
    (
        "Honey, we need to enrich our data, and for that, I need a DataEnrichmentObjective. We're interested in targets 'DE001', 'DE002', and 'DE003', using sensors 'RME40' and 'LMNT41'. "
        "This is 'S' level stuff, REAL mode, and use RATE_TRACK because we need that specific type for this, not the default. "
        "I only want to observe a maximum of 5 RSOs at a time, but I expect 8.0 revisits per hour for those chosen few. "
        "Get this started on 2026-01-01 09:00:00+00:00. Visibility check should be true, because we don't waste time on things we can't see. Priority is 18. "
        "Binning can be null, we don't need it here. And let's leave the objective end time open for now."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["DE001", "DE002", "DE003"],
        "sensor_name_list": ["RME40", "LMNT41"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 1, 1, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 18,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,  # Default
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 188: SensorCheckoutObjective
    (
        "Alright, gorgeous, it's time for a SensorCheckoutObjective. We need to make sure sensor 'RME55' is in tip-top shape. Classification is 'U', and this is a REAL mode checkout. "
        "We're focusing on the LEO orbital regime for this test. Tracking type is SIDEREAL. Priority? Let's make it a sassy 6. "
        "I want 2.0 revisits per hour. Objective starts on 2026-02-14 12:00:00+00:00 because even sensors need some Valentine's love. "
        "Visibility check is absolutely true. Patience is 20 minutes. Give me 6 frames per intent, and an integration time of 1.0 second. "
        "No intent start or end time, let the system figure it out. Binning? Not for this checkout, darling."
    ): {
        "classification_marking": "U",
        "sensor_name": "RME55",
        "orbital_regime": "LEO",
        "data_mode": "REAL",  # Default
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 20,
        "revisits_per_hour": 2.0,
        "number_of_frames": 6,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 14, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 6,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 189: BaselineAutonomyObjective
    (
        "Okay, queen, let's set up a BaselineAutonomyObjective. Assign it the UUID 'abcdef01-2345-6789-abcd-ef0123456789' right off the bat. "
        "Markings are 'U', as standard. Data mode is REAL, frame type LIGHT, and let's give this workhorse a priority of 1500 - it's important but not *the* most important, you know? "
        "For the baseline autonomy RSOs, I want you to track these catalog IDs: '25544' and '27435'. "
        "There's no end time; this baby runs until we say stop. And just for kicks, initially, let's list RSO IDs 'SAT123' and 'SAT456', though we know the system will update these. This is crucial, so make sure it's perfect!"
    ): {
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "classification_marking": "U",  # Default
        "data_mode": "REAL",  # Default
        "frame_type": "LIGHT",  # Default
        "priority": 1500,  # Default
        "baseline_autonomy_rso": "25544,27435",
        "objective_end_time": None,  # Default
        "rso_id_list": ["SAT123", "SAT456"],
        "objective_name": "BaselineAutonomyObjective",  # Default
    },
    # New Example 190: CatalogMaintenanceObjective
    (
        "Darling, let's create a CatalogMaintenanceObjective using sensors 'FANCYPANTS01' and 'GLITTERBEAM03'. "
        "We need this one marked with 'S' classification, running in 'REAL' data mode, and set the priority to a rather important '5'. "
        "Give it a patience of '15' minutes and an end time offset of '30' minutes from the intent's start, with the 'visibility_check' set to 'true' because we only want to see the stars that want to be seen! "
        "The objective itself should start at '2025-06-01 10:00:00+00:00' and wrap up by '2025-06-01 14:00:00+00:00'. "
        "Let's use 'RATE_TRACK' for tracking, focusing on the 'MEO' orbital regime. "
        "Include the following RSO IDs: '98765', '43210', and '55555'. That should keep things interesting!"
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 5,
        "rso_id_list": ["98765", "43210", "55555"],
        "sensor_name_list": ["FANCYPANTS01", "GLITTERBEAM03"],
        "visibility_check": True,
    },
    # New Example 191: SearchObjective
    (
        "Okay, sweetie, let's set up a SearchObjective for target 'TARGET007' using our trusty sensor 'SHARPEYE02'. "
        "This needs a 'C' classification marking and should be in 'SIMULATED' data mode for practice. "
        "Set the priority to a sensible '8' and let's use the 'SIDEREAL' tracking type. "
        "The whole objective runs from '2025-06-10 20:00:00+00:00' to '2025-06-10 23:00:00+00:00'. "
        "For the search parameters, use an 'initial_offset' of '90' seconds and a 'final_offset' of '120' seconds. "
        "We want a generous 'frame_overlap_percentage' of '0.85', and set the 'end_time_offset_minutes' to '60'. "
        "The 'search_type' should be 'CROSS_TRACK', and the search itself should kick off '20' minutes after the objective begins. We don't need a visibility check for this simulation."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 60,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 6, 10, 23, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 8,
        "search_start_time": "datetime.datetime(2025, 6, 10, 20, 20, tzinfo=TzInfo(UTC))",  # 20 minutes after start
        "search_type": "CROSS_TRACK",
        "sensor_name": "SHARPEYE02",
        "target_id": "TARGET007",
        "visibility_check": False,
    },
    # New Example 192: GeodssRevisitObjective
    (
        "For our GeodssRevisitObjective, we're going after targets 'GEMINI1', 'TAURUS2', and 'CANCER3' using sensors 'STELLARVIEW07' and 'COSMICEYE09'. "
        "This is a 'TS' classification, honey, and it's in 'REAL' data mode. "
        "Priority is a solid '10', and the tracking type is 'RATE_TRACK_SIDEREAL'. "
        "We'll start the objective at '2025-07-04 21:00:00+00:00'. "
        "Configure the camera with a 'readout_rate' of '0' (that's 1MHz, darling), a 'gain_setting' of '1' (Low Gain), and use 'soi_filter' position '2' (10% Light). "
        "Set 'auto_track_type' to '2' (Manual) with 'auto_track_roi_position' at '1' (PMT Boresite). "
        "Camera 'camera_mode' is '1' (Zoomed EBS), 'array_kind' is '1' (Photometer), 'binning_mode' is '0' (No Binning), and 'scan_mode' is '0' (Continuous). "
        "We'll skip '2' frames initially and make '3' observations per intent. No need to ignore other objectives right now."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 1,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 2,
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 4, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 30,  # Default
        "priority": 10,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["STELLARVIEW07", "COSMICEYE09"],
        "soi_filter_position": 2,
        "target_id_list": ["GEMINI1", "TAURUS2", "CANCER3"],
        "visibility_check": False,  # Default
    },
    # New Example 193: PeriodicRevisitObjective
    (
        "Let's schedule a PeriodicRevisitObjective for targets 'ORION45', 'CYGNUS67', and 'LYRA89'. "
        "We'll use sensor 'NIGHTHAWK11'. This is a 'U//FOUO' marking, darling, and we'll run it in 'EXERCISE' data mode. "
        "Set the priority to a lower '15' for this exercise. Patience is '20' minutes. "
        "We want to ignore other objective intent submissions because this is a dedicated exercise. "
        "The objective starts at '2025-08-20 15:30:00+00:00'. "
        "We need '7' frames per intent with an 'integration_time' of '3.5' seconds. "
        "Set the 'optimal_frames_per_hour' to a zippy '500'. We'll aim for '8.0' revisits per hour. Use 'RATE_TRACK_SIDEREAL' tracking."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["ORION45", "CYGNUS67", "LYRA89"],
        "sensor_name_list": ["NIGHTHAWK11"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 8, 20, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 194: UctObservationObjective
    (
        "Time for a UctObservationObjective, focusing on UCT RSOs 'UCT101', 'UCT202', and 'UCT303' with sensors 'SKYSCANNER05' and 'ORBITALEYE08'. "
        "This requires a 'U' classification marking and should be in 'REAL' data mode. "
        "We're targeting the 'GEO' orbital regime. Priority '12', please, with 'revisits_per_hour' set to '4.5'. "
        "Kick off this objective at '2025-09-05 22:00:00+00:00'. "
        "Let's enable 'sorting by brightest UCT' because we want to see those shining stars first! "
        "Set the 'end time offset' to '90' minutes and make sure 'visibility_check' is 'true'. "
        "We need '6' frames with an 'integration_time' of '1.8' seconds. Use 'RATE_TRACK_SIDEREAL' tracking."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["UCT101", "UCT202", "UCT303"],
        "sensor_name_list": ["SKYSCANNER05", "ORBITALEYE08"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 4.5,
        "number_of_frames": 6,
        "integration_time": 1.8,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 12,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 195: SingleIntentObjective
    (
        "Create a SingleIntentObjective, darling, for target ID 'SOLARIS4' and RSO ID 'LUNARIS9'. "
        "We'll use just one sensor this time: 'STARGAZER10'. "
        "This is a 'C' marking, in 'REAL' data mode, with 'RATE_TRACK' tracking. "
        "Priority is a sharp '7'. "
        "The objective should commence at '2025-10-15 09:00:00+00:00'. "
        "Let's capture '10' frames with an 'integration_time' of '4.0' seconds. "
        "Apply a 'binning' setting of '4'. No need for objective end time, this is a one-shot deal."
    ): {
        "classification_marking": "C",
        "target_id": "SOLARIS4",
        "rso_id": "LUNARIS9",
        "sensor_name_list": ["STARGAZER10"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 10,
        "integration_time": 4.0,
        "priority": 7,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 15, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 196: DataEnrichmentObjective
    (
        "Now, for a DataEnrichmentObjective focusing on targets 'DE_TARGET_A', 'DE_TARGET_B', and 'DE_TARGET_C' using sensors 'ENRICHMENTSCANNER01' and 'KNOWLEDGESEEKER02'. "
        "Mark this with 'S' classification and run it in 'TEST' data mode. "
        "We want 'RATE_TRACK_SIDEREAL' tracking. Set the 'max_rso_to_observe' to '10' and aim for '15' revisits per hour. "
        "The objective starts at '2025-11-01 14:00:00+00:00'. "
        "Let's set 'visibility_check' to 'true' to ensure we only try for observable RSOs. Priority '25', please."
    ): {
        "classification_marking": "S",
        "data_mode": "TEST",
        "objective_uuid": None,
        "target_id_list": ["DE_TARGET_A", "DE_TARGET_B", "DE_TARGET_C"],
        "sensor_name_list": ["ENRICHMENTSCANNER01", "KNOWLEDGESEEKER02"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 197: SensorCheckoutObjective
    (
        "Let's do a SensorCheckoutObjective for our brand new sensor, 'NEWVIEWFINDER01'. "
        "This is classified 'U//FOUO', naturally, and should be in 'REAL' data mode. "
        "We're checking its capabilities in the 'XGEO' orbital regime using 'RATE_TRACK_SIDEREAL' tracking. "
        "Priority '5', because we need this checked out promptly! "
        "We want '2.0' revisits per hour, starting the objective at '2025-12-10 18:00:00+00:00'. "
        "Set 'visibility_check' to 'true', give it a 'patience_minutes' of '45'. "
        "We need '8' frames per intent with an 'integration_time' of '5.0' seconds. This test is crucial!"
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "NEWVIEWFINDER01",
        "orbital_regime": "XGEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 2.0,
        "number_of_frames": 8,
        "integration_time": 5.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 10, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 198: BaselineAutonomyObjective
    (
        "Finally, let's establish a BaselineAutonomyObjective with the UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. "
        "This is classified 'U', operates in 'REAL' data mode, using the 'LIGHT' frame type. "
        "Give it a low priority of '2000', as it's just for baseline operations. "
        "We need it to continuously run, so no objective end time is needed. "
        "Include catalog IDs 'CATA1' and 'CATA2' for baseline tracking. "
        "The initial RSO ID list includes 'RSOX1', 'RSOY2', and 'RSOZ3', though we know this will be updated. "
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 2000,
        "baseline_autonomy_rso": "CATA1,CATA2",
        "objective_end_time": None,
        "rso_id_list": ["RSOX1", "RSOY2", "RSOZ3"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 199: CatalogMaintenanceObjective
    (
        "Cap'n needs a CatalogMaintenanceObjective to keep tabs on some space debris, those pesky RSO IDs be '77889' and '99001'. We'll be usin' our trusty spyglasses, SKYEYE-7 and LENSMASTER-3, for this important task. "
        "Mark all findings with U//FOUO, and run this as an EXERCISE, so no real cannonballs are fired, savvy? "
        "Set the priority to a lowly 50, as there be other treasures to hunt. Let the system be patient for 15 minutes before givin' up on a target. "
        "The end time offset should be a good 30 minutes past what's scheduled. This grand operation needs to run from 2025-10-10 10:00:00+00:00 until 2025-10-10 14:00:00+00:00, sharp! "
        "We be sailin' in the MEO regime for this venture. And listen well, don't ye be checkin' for visibility, we trust our charts. "
        "We'll use the RATE_TRACK tracking method, and be sure to set the binning to 2."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "end_time_offset_minutes": 30,
        "priority": 50,
        "sensor_name_list": ["SKYEYE-7", "LENSMASTER-3"],
        "rso_id_list": ["77889", "99001"],
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": 2,
        "visibility_check": False,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 200: SearchObjective
    (
        "Ahoy! We be needin' a SearchObjective, me lads, to find a lost treasure! The target ID for this elusive prize is 'target_alpha_789'. Ye shall use the keen 'HAWKEYE-9' spyglass. "
        "This be a top-secret treasure hunt, so mark it TS, and keep it under yer hats! We're in REAL mode this time, an actual voyage. "
        "Priority is numero uno (that's 1, ye scallywags!), for this be most urgent. For tracking, we'll employ the SIDEREAL method, following the stars. "
        "Set sail (start the objective) at 2025-11-05 14:00:00+00:00 and drop anchor (end the objective) at 2025-11-05 18:30:00+00:00. "
        "The initial offset before the suspected location is 45 seconds, and the final offset after it is 75 seconds. We need a frame overlap of 60 percent, so we don't miss a single doubloon. "
        "The end time offset for scheduling the intent will be 50 minutes. Our search pattern will be CROSS_TRACK, like a proper treasure map. "
        "The search itself, the real diggin', commences precisely 20 minutes after the objective begins. We aim to take 10 frames, with an integration time of 3.5 seconds for each to soak in the starlight."
    ): {
        "classification_marking": "TS",
        "target_id": "target_alpha_789",
        "sensor_name": "HAWKEYE-9",
        "search_type": "CROSS_TRACK",
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 50,
        "priority": 1,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 5, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 5, 18, 30, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 3.5,
        "search_start_time": "datetime.datetime(2025, 11, 5, 14, 20, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 20 minutes
        "objective_name": "SearchObjective",
    },
    # New Example 201: GeodssRevisitObjective
    (
        "Shiver me timbers! Craft a GeodssRevisitObjective for a couple o' specific celestial bodies we're trackin', known by their target IDs 'sat_beta_002' and 'debris_gamma_003'. "
        "Our advanced instruments for this particular venture be the 'GEOSCAN-A' and its twin, 'GEOSCAN-B'. Mark this one 'C' for Confidential, as it's sensitive information. "
        "This is a REAL operation, with a priority of 15. We'll stick to the standard RATE_TRACK_SIDEREAL for our tracking needs. "
        "The captain's logbook entry for the start time is precisely 2025-12-01 08:00:00+00:00. For the detailed settings: set the readout_rate to 0 (that's 1MHz, slow and steady captures more detail), "
        "the gain_setting to 1 (Low Gain, for those faint, distant objects), and the soi_filter to 2 (10% Light, we don't want to blind our delicate instruments!). "
        "The auto_track_type will be 2 (Manual, our seasoned gunners trust their own eyes and hands), camera_mode shall be 1 (Zoomed EBS, to get a real close look!), "
        "array_kind is 1 (Photometer, specially for measuring the shiny bits of treasure... or debris), binning_mode will be 0 (No Binning, for raw data), and the scan_mode is 0 (Continuous, keep those observations rollin' in!). "
        "Assign this grand objective the UUID of 'pirate-geodss-001'. No need for an objective end time to be specified now, let it run its course as needed."
    ): {
        "classification_marking": "C",
        "target_id_list": ["sat_beta_002", "debris_gamma_003"],
        "sensor_name_list": ["GEOSCAN-A", "GEOSCAN-B"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default
        "patience_minutes": 20,  # Custom, example used 30
        "revisits_per_hour": None,
        "number_of_frames": 3,  # Custom
        "integration_time": 1.0,  # Custom
        "objective_uuid": "pirate-geodss-001",
        "objective_start_time": "datetime.datetime(2025, 12, 1, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,  # Default
        "optimal_frames_per_hour": 300,  # Custom, example used 400
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
    # New Example 202: PeriodicRevisitObjective
    (
        "Avast ye! The quartermaster demands a PeriodicRevisitObjective, and quickly! We be trackin' targets 'comet_delta_404' and the space rock 'rock_epsilon_505'. "
        "For this, deploy the mighty 'STARGAZER-PRO' and the ever-watchful 'SKYWATCH-ULTRA' sensors. The markings for this operation are 'U', nice and simple. "
        "We'll run this in 'SIMULATED' mode, a dry run before the real plunder. Set the priority to a middlin' 5. Give it a patience of 20 minutes. "
        "The heart of the matter: we want 3 revisits per hour for each of these targets. This important quest begins at the stroke of midnight, 2026-01-15 00:00:00+00:00. "
        "For now, there's no set end time; let it run 'til further notice. With each revisit, we're looking to capture 8 frames, and each frame needs an integration time of 1.5 seconds. "
        "Set the binning at 4x4, groupin' those pixels. And a special instruction: don't ye be botherin' with other objectives' submissions; ignore 'em entirely! "
        "The optimal frames per hour can stay at the usual 400, that's a good rate."
    ): {
        "classification_marking": "U",
        "target_id_list": ["comet_delta_404", "rock_epsilon_505"],
        "sensor_name_list": ["STARGAZER-PRO", "SKYWATCH-ULTRA"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 20,
        "revisits_per_hour": 3.0,
        "number_of_frames": 8,
        "integration_time": 1.5,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 203: UctObservationObjective
    (
        "Heave ho, me hearties! We need a UctObservationObjective to track some Unidentified Floating Objects... I mean, Uncorrelated Tracks! These be mysterious signals in the void. "
        "The RSO IDs for these UCTs are 'uct_zeta_111' and the elusive 'uct_eta_222'. We shall assign our best trackers, the 'TRACKMASTER-5000' and 'FINDSAT-X1' sensors, to this task. "
        "Mark this mission 'S' for Secret, and conduct it in REAL mode. Our operations will be in the GEO regime, high above the seas. Priority is a high 8 for this sensitive watch. "
        "We aim for 5 revisits per hour, keepin' a close eye. This crucial mission kicks off on 2026-02-20 at 10:30:00+00:00. "
        "And mark me words, visibility check must be true for this one; we can't be chasin' ghosts! The end time offset for scheduling intents shall be a generous 75 minutes. "
        "During each observation, we'll collect 6 frames, and each frame gets an integration time of 2.5 seconds to gather enough light. "
        "A key strategy, me lads: sort 'em by the brightest UCT first! Show me the shiniest treasure out there!"
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["uct_zeta_111", "uct_eta_222"],
        "sensor_name_list": ["TRACKMASTER-5000", "FINDSAT-X1"],
        "data_mode": "REAL",  # Default
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,  # Default
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 5.0,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 10, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 204: SingleIntentObjective
    (
        "Yo ho ho, and a bottle of... data! A SingleIntentObjective is on the manifest today. The primary target ID is 'single_target_theta_333', and its associated RSO ID is 'rso_kappa_444'. "
        "For this focused endeavor, our chosen instruments are the precise 'ONEVIEW-MK1' and the steadfast 'PINPOINT-GZ'. The markings for this mission will be 'U//FOUO'. "
        "We'll be operating in 'TEST' mode for this. Set the tracking mechanism to RATE_TRACK, as we need to follow its specific path. Priority is 12 for this one, a standard task. "
        "The objective itself is to commence at 2026-03-10 15:00:00+00:00. During this observation, we're looking to grab 3 frames of data. "
        "Each frame should have an integration time of 4 seconds. And make sure to use binning set to 1, for no pixel grouping. "
        "Now, for the specific timing of the intent: it should start no earlier than 2026-03-10 15:15:00+00:00 and must end no later than 2026-03-10 16:00:00+00:00. Be precise, like a navigator's chart!"
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "single_target_theta_333",
        "rso_id": "rso_kappa_444",
        "sensor_name_list": ["ONEVIEW-MK1", "PINPOINT-GZ"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 3,
        "integration_time": 4.0,
        "priority": 12,
        "binning": 1,
        "intent_start_time": "datetime.datetime(2026, 3, 10, 15, 15, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 3, 10, 16, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 15, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 205: DataEnrichmentObjective
    (
        "Ahoy there, data-wranglers! Let's set up a DataEnrichmentObjective to make our treasure maps more detailed. We're particularly interested in targets identified as 'data_point_lambda_77', 'info_mu_88', and the valuable 'knowledge_nu_99'. "
        "The sensors we'll deploy for this task are the 'DATASCOOP-PRO' and the 'INFOGATHER-X'. Classification for this data haul is 'C' (Confidential). We're operating in REAL mode, gathering actual intel. "
        "For tracking, use the SIDEREAL method for this one, following the ancient star paths. We want to observe a maximum of 7 RSOs during this operation. "
        "Our goal is to achieve 9 revisits per hour for these targets, collecting as much data as possible. This important task officially starts on 2026-04-05 12:00:00+00:00. "
        "And it's crucial, make absolutely sure that the visibility check is active (set to true)! Priority for this enrichment mission is a standard 20. "
        "Let's assign this objective the unique identifier 'enrich-data-007'. The objective itself can run until we decide its rich enough."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": "enrich-data-007",
        "target_id_list": ["data_point_lambda_77", "info_mu_88", "knowledge_nu_99"],
        "sensor_name_list": ["DATASCOOP-PRO", "INFOGATHER-X"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 7,
        "revisits_per_hour": 9.0,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,  # Default
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,  # Default
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 206: SensorCheckoutObjective
    (
        "Alright, ye swabs, heave to! Time for a SensorCheckoutObjective. We need to give the new 'TESTBED-SENSOR-007' a proper shakedown before it joins the fleet. "
        "The classification marking for this checkout procedure is 'U'. Data mode will be 'EXERCISE', as we're just testing its capabilities. "
        "This sensor is designed primarily for the XGEO regime, way out in the deep dark. We'll use the standard RATE_TRACK_SIDEREAL for tracking during the test. "
        "Priority for this checkout will be 7. We're aiming for 2 revisits per hour on our test targets. The checkout procedure is scheduled to start at 2026-05-01 09:00:00+00:00. "
        "Visibility check must be true; we need to see what it's seeing! Give it a patience of 25 minutes before deciding a test step has failed. "
        "During each test sequence, we need to capture 10 frames. Set the integration time for each frame to a brief 0.5 seconds. Binning should be set to 2x2 for this checkout."
    ): {
        "classification_marking": "U",
        "sensor_name": "TESTBED-SENSOR-007",
        "orbital_regime": "XGEO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 25,
        "revisits_per_hour": 2.0,
        "number_of_frames": 10,
        "integration_time": 0.5,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 7,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 207: BaselineAutonomyObjective
    (
        "Listen up, ye scallywags, this be an important decree! We're initiating a BaselineAutonomyObjective, a long-term surveillance mission. "
        "The unique identifier for this grand and continuous endeavor shall be 'baseline-autonomy-jolly-roger-001'. The classification markings for all data gathered under this objective will be 'S' (Secret). "
        "Data mode is 'REAL', as we're observing the actual state of the cosmos. For this specific objective, the frame type will be 'DARK', for calibration purposes alongside light frames from other tasks. "
        "Priority is the highest it can be, 1500, as this underpins all our autonomous operations! The baseline autonomy RSO list, these be the critical catalog IDs we must always track, are 'cat_id_alpha', 'cat_id_beta', and 'cat_id_gamma'. "
        "This objective runs continuously, like the tides, so there's no end time. The RSO ID list that will be dynamically populated by the autonomy system includes initial placeholders such as 'rso_placeholder_1' and 'rso_placeholder_2', which will be updated in real-time."
    ): {
        "objective_uuid": "baseline-autonomy-jolly-roger-001",
        "classification_marking": "S",
        "data_mode": "REAL",  # Default
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "cat_id_alpha,cat_id_beta,cat_id_gamma",
        "objective_end_time": None,  # Default
        "rso_id_list": ["rso_placeholder_1", "rso_placeholder_2"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 208: CatalogMaintenanceObjective
    (
        "Create a CatalogMaintenanceObjective for sensors 'KrakenEye' and 'SaltySpyglass' carryin' "
        "C markings, SIMULATED mode, priority 50, patience of 15 mins, end time offset of 30 mins, "
        "and set visibility check to true, arrr. This voyage begins at 2025-06-01 01:00:00+00:00 "
        "and wraps up at 2025-06-01 06:00:00+00:00. We'll be usin' SIDEREAL trackin' in the MEO regime, "
        "and the list o' RSO IDs be '98765,43210,10987'. No special binning for this run."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 1, 6, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 1, 1, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 50,
        "rso_id_list": ["98765", "43210", "10987"],
        "sensor_name_list": ["KrakenEye", "SaltySpyglass"],
        "visibility_check": True,
    },
    # New Example 209: SearchObjective
    (
        "Set sail on a SearchObjective for target 'GhostShip42' usin' the 'WobblyLens' sensor, ye hear? "
        "Mark this with S, run in EXERCISE mode, and clap on priority 3. "
        "We'll employ RATE_TRACK for this pursuit. Commence the objective at 2025-06-15 14:00:00+00:00 "
        "and let it run until 2025-06-15 15:30:00+00:00. The initial offset be 90 seconds fore and the final offset 120 seconds aft, "
        "with a frame overlap o' 80%. The end time offset should be 50 minutes. "
        "This search will be an ALONG_TRACK type, startin' 20 minutes after the objective begins, and mind ye, do a visibility check!"
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 50,
        "final_offset": 120,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 6, 15, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 6, 15, 14, 20, tzinfo=TzInfo(UTC))",  # 20 mins after obj start
        "search_type": "ALONG_TRACK",
        "sensor_name": "WobblyLens",
        "target_id": "GhostShip42",
        "visibility_check": True,
    },
    # New Example 210: GeodssRevisitObjective
    (
        "We need a GeodssRevisitObjective for targets 'Seahorse1', 'Barnacle2', 'TidalWisp' usin' sensors 'DeepGazer' and 'SkyWatcher'. "
        "Mark it TS, sail in REAL mode, priority 8, and use RATE_TRACK_SIDEREAL. The objective is to commence at 2025-07-04 20:00:00+00:00. "
        "Set the readout_rate to 0 (1MHz), gain_setting to 1 (Low Gain), soi_filter to 2 (10% Light), auto_track_type to 2 (Manual), "
        "camera_mode to 1 (Zoomed EBS), array_kind to 1 (Photometer), binning_mode to 0 (No Binning), and scan_mode to 0 (Continuous). "
        "Also, set acquisition_type to 2 (Auto Rate Track), num_observations to 3, and num_skip_frames to 2. We'll ignore other objective submissions."
    ): {
        "acquisition_type": 2,
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Using default as not specified
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Using default as not specified
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": True,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 2,
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 4, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Using default as not specified
        "optimal_frames_per_hour": 400,  # Using default as not specified
        "overscan": 0,  # Using default as not specified
        "patience_minutes": 30,  # Using default as not specified
        "priority": 8,
        "rate_track_verify": 0,  # Using default as not specified
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["DeepGazer", "SkyWatcher"],
        "soi_filter_position": 2,
        "target_id_list": ["Seahorse1", "Barnacle2", "TidalWisp"],
        "visibility_check": False,  # Using default as not specified
    },
    # New Example 211: PeriodicRevisitObjective
    (
        "Chart a course for a PeriodicRevisitObjective for targets 'Starfish3', 'CoralReef', 'KelpForest' "
        "usin' sensors 'NorthStar', 'SouthernCross'. Set the classification marking to U//FOUO, "
        "sail in REAL mode, and give it priority 15. We'll allow a patience of 45 minutes "
        "and make sure we *do* worry about previous intent submissions from other objectives. "
        "Start this objective at 2025-08-20 10:00:00+00:00. We aim for 350 optimal frames per hour, "
        "request 8 frames per intent, and set the integration time to 3 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["Starfish3", "CoralReef", "KelpForest"],
        "sensor_name_list": ["NorthStar", "SouthernCross"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Using default as not specified
        "frame_type": "LIGHT",  # Using default as not specified
        "visibility_check": False,  # Using default as not specified
        "patience_minutes": 45,
        "revisits_per_hour": None,
        "number_of_frames": 8,
        "integration_time": 3.0,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 8, 20, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 212: UctObservationObjective
    (
        "Initiate a UctObservationObjective for UCT RSOs 'MermaidsTear', 'ShipwreckAlloy' "
        "usin' the 'DeepScanner' and 'VoidTracker' sensors. Mark this with U, operate in TEST mode, "
        "in the XGEO regime, and assign priority 7. We need 10 revisits per hour, "
        "and the mission starts at 2025-09-05 22:00:00+00:00. Be sure to enable sorting by the brightest UCTs, "
        "set the end time offset to 90 minutes, and perform a visibility check, savvy? "
        "Each intent should have 7 frames with an integration time of 1.5 seconds."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["MermaidsTear", "ShipwreckAlloy"],
        "sensor_name_list": ["DeepScanner", "VoidTracker"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Using default as not specified
        "frame_type": "LIGHT",  # Using default as not specified
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Using default as not specified
        "revisits_per_hour": 10.0,
        "number_of_frames": 7,
        "integration_time": 1.5,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 213: SingleIntentObjective
    (
        "Plot a course for a SingleIntentObjective with target ID 'LostBooty' and RSO ID 'GoldenDoubloon'. "
        "Employ sensors 'MastEye' and 'CrowNestCam' for this task. "
        "Set the classification marking to S, run in SIMULATED mode, "
        "usin' SIDEREAL trackin', and assign priority 1. "
        "The objective should commence at 2025-10-10 16:30:00+00:00. "
        "We require 10 frames for this intent, each with an integration time of 0.5 seconds, "
        "and set the binning to 4."
    ): {
        "classification_marking": "S",
        "target_id": "LostBooty",
        "rso_id": "GoldenDoubloon",
        "sensor_name_list": ["MastEye", "CrowNestCam"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Using default as not specified
        "number_of_frames": 10,
        "integration_time": 0.5,
        "priority": 1,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 16, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 214: DataEnrichmentObjective
    (
        "Embark on a DataEnrichmentObjective for targets 'AncientMap', 'SunkenChest', 'KrakenInk', 'GhostLight'. "
        "Utilize sensors 'OceanSurveyor', 'SkyProspector'. "
        "Mark this objective C, run in EXERCISE mode, employin' RATE_TRACK tracking. "
        "Set the maximum RSO to observe to 12, and aim for 15 revisits per hour. "
        "This venture starts at 2025-11-11 04:00:00+00:00, and remember to check for visibility!"
    ): {
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["AncientMap", "SunkenChest", "KrakenInk", "GhostLight"],
        "sensor_name_list": ["OceanSurveyor", "SkyProspector"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Using default as not specified
        "binning": None,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 11, 11, 4, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,  # Using default as not specified
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 215: SensorCheckoutObjective
    (
        "Conduct a SensorCheckoutObjective usin' classification_marking='TS' and sensor_name='WeatherSail'. "
        "Set data_mode='TEST', orbital_regime='LEO', collect_request_type='RATE_TRACK', priority=5. "
        "Request 2.5 revisits per hour. The objective starts at '2025-12-01 10:00:00+00:00', "
        "and we require a visibility check, arrr. Allow a patience of 60 minutes, "
        "with 3 frames per intent and an integration time of 4 seconds."
    ): {
        "classification_marking": "TS",
        "sensor_name": "WeatherSail",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Using default as not specified
        "visibility_check": True,
        "patience_minutes": 60,
        "revisits_per_hour": 2.5,
        "number_of_frames": 3,
        "integration_time": 4.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 216: BaselineAutonomyObjective
    (
        "Hoist the flag for a BaselineAutonomyObjective with UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. "
        "Fly the C markings, operate in REAL mode, with DARK frame type, and set a high priority of 10. "
        "We'll be trackin' baseline autonomy RSOs with catalog IDs '24680' and '13579'. "
        "Include the following RSO IDs in the list: '98765', '21098', and '43210'. "
        "This mission will have an end time o' '2026-01-01 00:00:00+00:00'."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "C",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 10,
        "baseline_autonomy_rso": "24680,13579",
        "objective_end_time": "datetime.datetime(2026, 1, 1, 0, 0, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["98765", "21098", "43210"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 217: CatalogMaintenanceObjective
    (
        "Ahoy there! I be needin' a CatalogMaintenanceObjective for these fine sensors RME09 and LMNT08, "
        "with classification 'S', REAL mode, and a priority of 7. Set patience to 15 mins and end time offset to 30 mins. "
        "Make the visibility check true, arr! Start at 2025-06-15 14:30:00+00:00 and end at 2025-06-15 18:45:00+00:00. "
        "Use RATE_TRACK_SIDEREAL in the MEO regime. The RSO ID list should have '34512,78901,12345'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 18, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 7,
        "rso_id_list": ["34512", "78901", "12345"],
        "sensor_name_list": ["RME09", "LMNT08"],
        "visibility_check": True,
    },
    # New Example 218: SearchObjective
    (
        "Arr matey! Set up a SearchObjective for target 54321 with the sensor UKR15. Mark it 'U//FOUO', "
        "put it in TEST mode with priority 3. Use RATE_TRACK tracking. Starts at 2025-08-19 08:45:00+00:00 and "
        "ends at 2025-08-19 12:15:00+00:00. Initial offset be 45 seconds, final offset 75 seconds, and frame "
        "overlap at 65%. End time offset should be 50 minutes. Search type be CROSS_TRACK with search start time "
        "20 minutes after objective start. And don't ye forget to set visibility check to true!"
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "TEST",
        "end_time_offset_minutes": 50,
        "final_offset": 75,
        "frame_overlap_percentage": 0.65,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 8, 19, 12, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 19, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 8, 19, 9, 5, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "UKR15",
        "target_id": "54321",
        "visibility_check": True,
    },
    # New Example 219: GeodssRevisitObjective
    (
        "Yo-ho-ho! Configure a GeodssRevisitObjective for targets 23456,78912 with sensors RME21 and LMNT26. "
        "Classify it as 'C', use REAL mode, and SIDEREAL tracking, with priority 8. Start at 2025-09-05 22:15:00+00:00. "
        "Set readout_rate to 0 (1MHz), gain_setting to 1 (Low Gain), soi_filter to 2 (10% Light), and auto_track_type to 2 (Manual). "
        "Camera_mode be 1 (Zoomed EBS), array_kind be 1 (Photometer), binning_mode 0 (No Binning), and scan_mode 0 (Continuous). "
        "Make patience 25 minutes, me hearties!"
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
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
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 25,
        "priority": 8,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["RME21", "LMNT26"],
        "soi_filter_position": 2,
        "target_id_list": ["23456", "78912"],
        "visibility_check": False,
    },
    # New Example 220: PeriodicRevisitObjective
    (
        "Avast ye! Create a PeriodicRevisitObjective for targets 33221,55443,11998 using the fine sensors RME11 and LMNT12. "
        "Give it 'TS' classification, SIMULATED mode, and a priority of 5. Set patience to 45 minutes. "
        "Start the objective at 2025-07-04 03:15:00+00:00. Make it 3 revisits per hour, with 8 frames per intent "
        "and an integration time of 1.5 seconds. Set optimal frames per hour to 350, and don't ignore other objective submissions!"
    ): {
        "classification_marking": "TS",
        "target_id_list": ["33221", "55443", "11998"],
        "sensor_name_list": ["RME11", "LMNT12"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 45,
        "revisits_per_hour": 3.0,
        "number_of_frames": 8,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 7, 4, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 221: UctObservationObjective
    (
        "Yarr! I be needin' a UctObservationObjective for UCT RSOs 87654,12399,76543 with sensors RME25 and LMNT28. "
        "Give it 'S' classification, EXERCISE mode, MEO regime, and priority 6. Set it to 8 revisits per hour. "
        "Start at 2025-10-10 13:40:00+00:00. Enable sorting by brightest UCT, set end time offset to 40 minutes, "
        "check visibility be true. Use 3 frames per intent with 2.5 second integration time. SIDEREAL tracking, if ye please!"
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["87654", "12399", "76543"],
        "sensor_name_list": ["RME25", "LMNT28"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 40,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 13, 40, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 6,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 222: SingleIntentObjective
    (
        "Shiver me timbers! Create a SingleIntentObjective with target ID 44556, RSO ID 99887, using sensors RME33 and LMNT36. "
        "Set 'C' classification, EXERCISE mode, and RATE_TRACK tracking with priority 4. Start your objective at "
        "2025-11-03 05:30:00+00:00. Make it 10 frames, 3 seconds integration time, and binning 4. "
        "This be a critical mission, so get it right or walk the plank!"
    ): {
        "classification_marking": "C",
        "target_id": "44556",
        "rso_id": "99887",
        "sensor_name_list": ["RME33", "LMNT36"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 10,
        "integration_time": 3,
        "priority": 4,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 3, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 223: DataEnrichmentObjective
    (
        "Ahoy there, me buckos! Set up a DataEnrichmentObjective for targets 77889, 11223, 44556, 78901 with sensors RME42 and LMNT44. "
        "Mark it 'U' classification, REAL mode, SIDEREAL tracking. Make max RSO to observe be 12, and set 15 revisits per hour. "
        "Start this venture at 2025-12-15 18:10:00+00:00 and end it at 2025-12-16 02:30:00+00:00. Set priority to 15 and keep "
        "visibility check enabled. This mission be critical for our treasure hunt!"
    ): {
        "classification_marking": "U",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["77889", "11223", "44556", "78901"],
        "sensor_name_list": ["RME42", "LMNT44"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 12, 15, 18, 10, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 16, 2, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 224: SensorCheckoutObjective
    (
        "Arr matey! Prepare a SensorCheckoutObjective with 'U//FOUO' classification and sensor_name 'RME08'. "
        "Use SIMULATED mode, LEO regime, and RATE_TRACK tracking type. Priority should be 8, with 2.5 revisits per hour. "
        "Begin at 2026-01-10 11:45:00+00:00 and be checking visibility. Set patience to 20 minutes, frames to 7, and "
        "integration time to 1.75 seconds. This be for calibratin' our new sensor, so make it shipshape!"
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "RME08",
        "orbital_regime": "LEO",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 2.5,
        "number_of_frames": 7,
        "integration_time": 1.75,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 11, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 8,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 225: BaselineAutonomyObjective
    (
        "Yo-ho-ho! Create a BaselineAutonomyObjective with UUID '789a4567-e11c-42d3-b999-426614174999'. "
        "Set 'S' classification, TEST mode, LIGHT frame type, and priority 2000. Use baseline autonomy RSO IDs '28180,39210,45678' "
        "and RSO ID list '22222,33334,44445'. This be an ongoing mission, so no end time be needed, savvy?"
    ): {
        "objective_uuid": "789a4567-e11c-42d3-b999-426614174999",
        "classification_marking": "S",
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "priority": 2000,
        "baseline_autonomy_rso": "28180,39210,45678",
        "objective_end_time": None,
        "rso_id_list": ["22222", "33334", "44445"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 228: CatalogMaintenanceObjective
    (
        "Alright, let's generate a CatalogMaintenanceObjective. We need this for sensors DELTA01 and ECHO05. Classification will be S. "
        "Set mode to SIMULATED. Priority is critical, set it to 5. Patience threshold will be 15 minutes, and the end time offset is 30 minutes. "
        "Visibility check must be enabled, true. This objective commences on 2025-11-15 08:00:00+00:00 and will conclude on 2025-11-15 12:30:00+00:00. "
        "Tracking type will be RATE_TRACK, operating in the MEO regime. The RSO ID list for this tasking is '55001,55002,55003'. Binning should be set to 2x2."
    ): {
        "binning": 2,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 11, 15, 12, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 11, 15, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 5,
        "rso_id_list": ["55001", "55002", "55003"],
        "sensor_name_list": ["DELTA01", "ECHO05"],
        "visibility_check": True,
    },
    # New Example 229: SearchObjective
    (
        "We need a SearchObjective established for target ID 'TGT-ALPHA-776'. The designated sensor for this operation is FOXTROT99. "
        "Mark this operation with U//FOUO. We're operating in REAL data mode. Priority is set to 3. "
        "The tracking methodology will be SIDEREAL. The operational window opens at 2025-12-01 03:00:00+00:00 and closes at 2025-12-01 05:00:00+00:00. "
        "Initial offset needs to be 45 seconds, with a final offset of 75 seconds. Ensure frame overlap is at 60 percent. The end time offset for intent scheduling is 35 minutes. "
        "The search type is CROSS_TRACK. The search itself will kick off precisely 10 minutes after the objective's start time. Specify 10 frames with an integration time of 1.5 seconds each."
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 1.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 12, 1, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 12, 1, 3, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 12, 1, 3, 10, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 10 minutes
        "search_type": "CROSS_TRACK",
        "sensor_name": "FOXTROT99",
        "target_id": "TGT-ALPHA-776",
        "visibility_check": False,
    },
    # New Example 230: GeodssRevisitObjective
    (
        "Initiate a GeodssRevisitObjective. Target IDs are 'GEO-SAT-001' and 'GEO-COM-002'. Sensors assigned are GOLF01 and HOTEL02. "
        "Classification marking is TS. Data mode is REAL. Priority level 7. Tracking type is RATE_TRACK_SIDEREAL. "
        "This objective starts on 2026-01-10 20:00:00+00:00. The readout rate is 0 (1MHz), gain setting 1 (Low Gain). "
        "SOI filter position is 2 (10% Light). Auto track type is 2 (Manual). Camera mode 1 (Zoomed EBS). Array kind is 0 (Main). "
        "Binning mode 0 (No Binning). Scan mode 0 (Continuous). Set number of observations to 3 and observation interval to 5.5 seconds. "
        "Patience is 45 minutes. Ignore other objective intent submissions. Optimal frames per hour: 300."
    ): {
        "acquisition_type": 1,  # Manual Sidereal, assuming based on auto_track_type and context, schema says (0=Auto Sidereal, 1=Manual Sidereal, 2=Auto Rate Track)
        "array_kind": 0,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": True,
        "integration_time": None,  # Not specified in prompt, using schema default
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 0,
        "number_of_frames": None,  # Not specified in prompt, using schema default
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2026, 1, 10, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 5.5,
        "optimal_frames_per_hour": 300,
        "overscan": 0,
        "patience_minutes": 45,
        "priority": 7,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,  # Not specified in prompt, using schema default
        "scan_mode": 0,
        "sensor_name_list": ["GOLF01", "HOTEL02"],
        "soi_filter_position": 2,
        "target_id_list": ["GEO-SAT-001", "GEO-COM-002"],
        "visibility_check": False,
    },
    # New Example 231: PeriodicRevisitObjective
    (
        "Draft a PeriodicRevisitObjective. Target IDs for this are 'PR-TGT-X1' and 'PR-TGT-Y2'. Utilize sensors INDIA07 and JULIET08. "
        "Marking is C. Mode is EXERCISE. Priority is 4. Patience will be 25 minutes. Do not ignore other objective submissions. "
        "This objective kicks off on 2026-02-20 10:00:00+00:00 and concludes 2026-02-21 10:00:00+00:00. "
        "We need 3 revisits per hour. Capture 7 frames per intent, with an integration time of 2.5 seconds. Binning will be 1x1. "
        "Optimal frames per hour should be set to 350. Visibility check is off. Intent start time is 2026-02-20 11:00:00+00:00 and intent end time is 2026-02-21 09:00:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id_list": ["PR-TGT-X1", "PR-TGT-Y2"],
        "sensor_name_list": ["INDIA07", "JULIET08"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 3.0,
        "number_of_frames": 7,
        "integration_time": 2.5,
        "binning": 1,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 2, 21, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 350,
        "objective_uuid": None,
        "intent_start_time": "datetime.datetime(2026, 2, 20, 11, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 2, 21, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 232: UctObservationObjective
    (
        "Prepare a UctObservationObjective. The UCT RSO IDs are 'UCT-RSO-1010', 'UCT-RSO-2020'. Sensors KILO11 and LIMA12 are on task. "
        "Classification is U. Data mode is TEST. This is for the XGEO regime. Priority is high, set to 2. "
        "Objective starts 2026-03-05 00:00:00+00:00. We need 8 revisits per hour. Sort by brightest UCT. "
        "End time offset is 90 minutes. Visibility check must be true. Patience is 20 minutes. "
        "Acquire 4 frames with an integration time of 3 seconds. No binning specified."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["UCT-RSO-1010", "UCT-RSO-2020"],
        "sensor_name_list": ["KILO11", "LIMA12"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 4,
        "integration_time": 3.0,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 5, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default
        "priority": 2,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 233: SingleIntentObjective
    (
        "Generate a SingleIntentObjective. Target ID is 'SI-TGT-A45'. RSO ID is 'RSO-XYZ-987'. Sensors MIKE14 and NOVEMBER15 are assigned. "
        "Classification: U//FOUO. Data mode: REAL. Tracking: RATE_TRACK. Priority: 1, this is urgent. "
        "Objective starts 2026-04-12 15:30:00+00:00. The objective will end on 2026-04-12 16:00:00+00:00. "
        "We require 10 frames, integration time of 0.5 seconds. Binning is 4x4. "
        "The intent should start at 2026-04-12 15:35:00+00:00 and end by 2026-04-12 15:55:00+00:00."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "SI-TGT-A45",
        "rso_id": "RSO-XYZ-987",
        "sensor_name_list": ["MIKE14", "NOVEMBER15"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 10,
        "integration_time": 0.5,
        "priority": 1,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2026, 4, 12, 15, 35, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 4, 12, 15, 55, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 12, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 4, 12, 16, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 234: DataEnrichmentObjective
    (
        "Construct a DataEnrichmentObjective. Target IDs: 'DE-RSO-01', 'DE-RSO-02', 'DE-RSO-03'. Sensors OSCAR20 and PAPA21 will handle this. "
        "Classification is S. Data mode is SIMULATED. Tracking will be SIDEREAL. "
        "Set max RSO to observe to 10. We need 15 revisits per hour. "
        "This objective starts on 2026-05-01 06:00:00+00:00 and has no specific end time, let it run. Priority is 15. "
        "Visibility check must be active. Binning is 2. Intent start time will be 30 minutes after objective start."
    ): {
        "classification_marking": "S",
        "data_mode": "SIMULATED",
        "objective_uuid": None,
        "target_id_list": ["DE-RSO-01", "DE-RSO-02", "DE-RSO-03"],
        "sensor_name_list": ["OSCAR20", "PAPA21"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": 2,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # For continuous running
        "priority": 15,
        "intent_start_time": "datetime.datetime(2026, 5, 1, 6, 30, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 30 mins
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 235: SensorCheckoutObjective
    (
        "Formulate a SensorCheckoutObjective. The sensor under scrutiny is QUEBEC55. "
        "Classification: U. Orbital regime: LEO. Data mode: TEST. Tracking: RATE_TRACK_SIDEREAL. Priority: 5. "
        "We need 2.5 revisits per hour. Objective is to start 2026-06-10 12:00:00+00:00. Visibility check: true. "
        "Patience: 40 minutes. Number of frames will be 8, integration time 1.2 seconds. Binning is 3. "
        "The objective will conclude on 2026-06-10 18:00:00+00:00. Intent start time should be immediate with objective start, and intent end time should match objective end time."
    ): {
        "classification_marking": "U",
        "sensor_name": "QUEBEC55",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.5,
        "number_of_frames": 8,
        "integration_time": 1.2,
        "binning": 3,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 6, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 6, 10, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": "datetime.datetime(2026, 6, 10, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 6, 10, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 236: BaselineAutonomyObjective
    (
        "Create a BaselineAutonomyObjective. The UUID for this operation is 'ba445e67-f54b-44d3-b456-426614175111'. "
        "Markings are U//FOUO. Data mode: EXERCISE. Frame type will be DARK. Priority is set to 500. "
        "Baseline autonomy RSO catalog IDs are 'CAT-ID-700', 'CAT-ID-800'. This objective has no end time; it runs continuously. "
        "RSO ID list will initially include 'RSO-BASE-007' and 'RSO-BASE-008', but this will be updated at runtime."
    ): {
        "objective_uuid": "ba445e67-f54b-44d3-b456-426614175111",
        "classification_marking": "U//FOUO",
        "data_mode": "EXERCISE",
        "frame_type": "DARK",
        "priority": 500,
        "baseline_autonomy_rso": "CAT-ID-700,CAT-ID-800",
        "objective_end_time": None,  # For continuous running
        "rso_id_list": ["RSO-BASE-007", "RSO-BASE-008"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 237: CatalogMaintenanceObjective
    (
        "Initiate a CatalogMaintenanceObjective for sensors ALPHA01 and BRAVO03 under S classification. "
        "Operate in REAL data mode with a priority of 500. "
        "Allow for a patience window of 15 minutes and set the end time offset to 30 minutes out. "
        "Require a visibility check. "
        "The operational window is from 2025-06-01 10:00:00+00:00 to 2025-06-01 18:00:00+00:00. "
        "Target LEO regime using SIDEREAL tracking. "
        "Include RSO IDs '98765' and '45632'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 1, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "LEO",
        "patience_minutes": 15,
        "priority": 500,
        "rso_id_list": ["98765", "45632"],
        "sensor_name_list": ["ALPHA01", "BRAVO03"],
        "visibility_check": True,
    },
    # New Example 238: SearchObjective
    (
        "Command a SearchObjective targeting RSO '24680' utilizing the DELTA05 sensor. "
        "This operation is classified TS and will run in REAL mode. "
        "Assign this a top priority of 1. "
        "Employ a RASTER search type with RATE_TRACK collection. "
        "Execution window begins at 2025-06-15 23:00:00+00:00 and concludes at 2025-06-16 03:00:00+00:00. "
        "Set the initial offset to 90 seconds and the final offset to 120 seconds. "
        "Ensure a frame overlap of 80 percent. "
        "The end time offset for tasking should be 35 minutes from the objective start. "
        "Search start time is automatically set 15 minutes post objective initiation."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 35,
        "final_offset": 120,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 6, 16, 3, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 23, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "search_start_time": "datetime.datetime(2025, 6, 15, 23, 15, tzinfo=TzInfo(UTC))",  # Calculated as start + 15 mins
        "search_type": "RASTER",
        "sensor_name": "DELTA05",
        "target_id": "24680",
        "visibility_check": False,
    },
    # New Example 239: GeodssRevisitObjective
    (
        "Execute a GeodssRevisitObjective for targets '112233' and '445566' using sensors GEO1, GEO2. "
        "Classification is U//FOUO, data mode REAL. "
        "Standard RATE_TRACK_SIDEREAL tracking applies. "
        "Initiate this objective at 2025-07-04 02:00:00+00:00. "
        "Configure the camera with readout rate 0 (1MHz) and gain setting 1 (Low Gain). "
        "Set the SOI filter to position 2 (10% Light), auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), "
        "array kind to 1 (Photometer), binning mode to 0 (No Binning), and scan mode to 0 (Continuous). "
        "Require 2 observations per intent."
    ): {
        "acquisition_type": 0,  # Default Auto Sidereal if not specified in prompt
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default Optical Boresite if not specified
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default Configure if not specified
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default if not specified
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 2,
        "num_skip_frames": 0,  # Default if not specified
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 4, 2, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default if not specified
        "optimal_frames_per_hour": 400,  # Default if not specified
        "overscan": 0,  # Default Normal if not specified
        "patience_minutes": 30,  # Default if not specified
        "priority": 10,  # Default if not specified
        "rate_track_verify": 0,  # Default No Verify if not specified
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEO1", "GEO2"],
        "soi_filter_position": 2,
        "target_id_list": ["112233", "445566"],
        "visibility_check": False,  # Default if not specified
    },
    # New Example 240: PeriodicRevisitObjective
    (
        "Submit a PeriodicRevisitObjective targeting RSOs '778899' and '001122' using sensor CHARLIE07. "
        "Apply U classification and operate in TEST data mode. "
        "Priority is set at 5. "
        "Patience is 45 minutes. "
        "Do not ignore other objective intent submissions. "
        "Begin this objective at 2025-07-20 21:00:00+00:00. "
        "Specify 10 frames per intent with an integration time of 3.5 seconds. "
        "Optimal frames per hour is 500."
    ): {
        "classification_marking": "U",
        "target_id_list": ["778899", "001122"],
        "sensor_name_list": ["CHARLIE07"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 45,
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 7, 20, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 241: UctObservationObjective
    (
        "Generate a UctObservationObjective for UCT RSOs '334455', '667788', '990011' via sensors LIMA11, MIKE13. "
        "This task is classified S and uses REAL data mode. "
        "Focus on the GEO orbital regime using RATE_TRACK tracking. "
        "Priority is set to 8. "
        "Aim for 10 revisits per hour. "
        "Start operations at 2025-08-05 14:00:00+00:00. "
        "Enable visibility check true. "
        "Set the end time offset to 90 minutes. "
        "Instruct the system to sort by brightest UCT. "
        "Collect 8 frames per intent with a 1.8 second integration time."
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["334455", "667788", "990011"],
        "sensor_name_list": ["LIMA11", "MIKE13"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 10.0,
        "number_of_frames": 8,
        "integration_time": 1.8,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 8, 5, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 242: SingleIntentObjective
    (
        "Submit a SingleIntentObjective for target ID 'TARGET456', associated with RSO ID 'RSO789', "
        "using sensor NOVEMBER14. "
        "This is a C classified mission in REAL mode. "
        "Utilize SIDEREAL tracking with a priority of 15. "
        "The objective starts at 2025-08-20 17:30:00+00:00. "
        "Request 1 frame with an integration time of 5 seconds and binning setting 4."
    ): {
        "classification_marking": "C",
        "target_id": "TARGET456",
        "rso_id": "RSO789",
        "sensor_name_list": ["NOVEMBER14"],
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 1,
        "integration_time": 5.0,
        "priority": 15,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 8, 20, 17, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 243: DataEnrichmentObjective
    (
        "Issue a DataEnrichmentObjective covering targets 'ALPHA7', 'BETA8', 'GAMMA9', 'DELTA10'. "
        "Assign sensors OSCAR16 and PAPA18 to this task. "
        "Mark this U//FOUO and set data mode to SIMULATED. "
        "Employ RATE_TRACK_SIDEREAL for collection requests. "
        "The objective begins at 2025-09-10 09:00:00+00:00. "
        "Limit observations to a maximum of 12 RSOs per revisit and schedule 15 revisits per hour. "
        "Disable the visibility check."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "SIMULATED",
        "objective_uuid": None,
        "target_id_list": ["ALPHA7", "BETA8", "GAMMA9", "DELTA10"],
        "sensor_name_list": ["OSCAR16", "PAPA18"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 9, 10, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,  # Default
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 244: SensorCheckoutObjective
    (
        "Conduct a SensorCheckoutObjective for sensor QUEBEC20. "
        "This is a U classification, using EXERCISE data mode. "
        "Focus on the MEO orbital regime. "
        "Use RATE_TRACK_SIDEREAL collection. "
        "Set priority to 25. "
        "Schedule 2.5 revisits per hour. "
        "Objective commencement is 2025-10-05 04:00:00+00:00. "
        "Visibility check is required. "
        "Patience is set at 20 minutes. "
        "Request 3 frames per intent with a 2.0 second integration time."
    ): {
        "classification_marking": "U",
        "sensor_name": "QUEBEC20",
        "orbital_regime": "MEO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 2.5,
        "number_of_frames": 3,
        "integration_time": 2.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 5, 4, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 25,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 245: BaselineAutonomyObjective
    (
        "Establish a BaselineAutonomyObjective with UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. "
        "Apply TS classification and operate in REAL data mode. "
        "Utilize LIGHT frame type. "
        "Set the priority to 5000. "
        "Specify baseline autonomy RSOs using catalog IDs 'CAT101' and 'CAT202'. "
        "Include RSO IDs 'RSOAAA', 'RSOBBB', 'RSOCCC'. "
        "This objective is to run continuously without a set end time."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "TS",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 5000,
        "baseline_autonomy_rso": "CAT101,CAT202",
        "objective_end_time": None,
        "rso_id_list": ["RSOAAA", "RSOBBB", "RSOCCC"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 246: CatalogMaintenanceObjective
    (
        "Establish a CatalogMaintenanceObjective mission with MEO orbital regime for tracking space objects. Implement C classification marking, REAL operational mode, priority level 15, and patience setting of 20 minutes. Deploy sensors AKRO8, BRNO3, and DNVR5 with standard RATE_TRACK_SIDEREAL tracking. Set end time offset to 30 minutes and enable visibility check. Mission commences at 2025-03-15 08:30:00+00:00 and concludes at 2025-03-15 14:45:00+00:00. Target RSO IDs are 34892, 55123, and 77901."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 3, 15, 14, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 3, 15, 8, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 20,
        "priority": 15,
        "rso_id_list": ["34892", "55123", "77901"],
        "sensor_name_list": ["AKRO8", "BRNO3", "DNVR5"],
        "visibility_check": True,
    },
    # New Example 247: SearchObjective
    (
        "Execute SearchObjective for target 98765 using sensor TMPL7. Implementation requires TS classification, EXERCISE data mode, and priority 3. Use CROSS_TRACK search pattern with SIDEREAL tracking method. Begin at 2025-04-10 03:15:00+00:00 and terminate at 2025-04-10 05:30:00+00:00. Set initial offset to 45 seconds and final offset to 120 seconds with 85% frame overlap. Configure end time offset to 35 minutes, number of frames to 8, and integration time to 3.5 seconds. Search start time is designated as 2025-04-10 04:00:00+00:00. Conduct operation with visibility check enabled."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 35,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 3.5,
        "number_of_frames": 8,
        "objective_end_time": "datetime.datetime(2025, 4, 10, 5, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 4, 10, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 4, 10, 4, 0, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "TMPL7",
        "target_id": "98765",
        "visibility_check": True,
    },
    # New Example 248: GeodssRevisitObjective
    (
        "Deploy GeodssRevisitObjective for targets 78542, 36914, and 22450 utilizing sensors GDSS2 and GDSS5. Maintain S classification marking with SIMULATED mode and priority 8. Configure with RATE_TRACK tracking and initiate at 2025-01-05 22:10:00+00:00. Set technical parameters as follows: readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), scan_mode 0 (Continuous). Establish patience minutes at 25 and revisits per hour at 3.5."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "command": 0,
        "data_mode": "SIMULATED",
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
        "objective_start_time": "datetime.datetime(2025, 1, 5, 22, 10, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 25,
        "priority": 8,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": 3.5,
        "scan_mode": 0,
        "sensor_name_list": ["GDSS2", "GDSS5"],
        "soi_filter_position": 2,
        "target_id_list": ["78542", "36914", "22450"],
        "visibility_check": False,
    },
    # New Example 249: PeriodicRevisitObjective
    (
        "Conduct PeriodicRevisitObjective for high-value targets 45239, 89125, and 33785 with U//FOUO classification marking. Utilize sensors HNCO4, SNTG2, and KVLN9 in SIMULATED mode with RATE_TRACK collection. Execute mission with priority 6, setting patience to 40 minutes and revisits per hour to 2.5. Configure for 10 frames per observation with 1.8 seconds integration time and binning value of 3. Operation commences at 2025-05-08 14:20:00+00:00 and terminates at 2025-05-08 23:45:00+00:00. Enable visibility check and allow objective to override existing intent submissions."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["45239", "89125", "33785"],
        "sensor_name_list": ["HNCO4", "SNTG2", "KVLN9"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.5,
        "number_of_frames": 10,
        "integration_time": 1.8,
        "binning": 3,
        "objective_start_time": "datetime.datetime(2025, 5, 8, 14, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 5, 8, 23, 45, tzinfo=TzInfo(UTC))",
        "priority": 6,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 250: UctObservationObjective
    (
        "Initiate critical UctObservationObjective for UCT objects 42391, 67829, and 95437 in XGEO regime. Deploy specialized sensors CYRT3 and HMTN8 with TS classification level. Apply SIDEREAL tracking method in EXERCISE mode with priority 7. Set revisits to 8 per hour and enable brightness sorting for maximum detection probability. Configure 12 frames per observation with 4.2 seconds integration time and no binning. Establish objective start at 2025-02-17 11:25:00+00:00 with end time offset of 90 minutes. Patience threshold is 35 minutes. Disable visibility checking for this specialized high-altitude observation."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["42391", "67829", "95437"],
        "sensor_name_list": ["CYRT3", "HMTN8"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": False,
        "patience_minutes": 35,
        "revisits_per_hour": 8.0,
        "number_of_frames": 12,
        "integration_time": 4.2,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 2, 17, 11, 25, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 251: SingleIntentObjective
    (
        "Authorize SingleIntentObjective for immediate high-priority collection against RSO 88532 using target ID 77294. Allocate sensors DKOT1, DMSC9, and PLOM6 with C classification and TEST mode. Configure with RATE_TRACK collection and 15 frames at 5.0 seconds integration time. Implement binning at level 4 and set priority to maximum level 2. Mission window established from 2025-04-22 07:00:00+00:00 to 2025-04-22 10:00:00+00:00 with intent submission required no later than 2025-04-22 06:30:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id": "77294",
        "rso_id": "88532",
        "sensor_name_list": ["DKOT1", "DMSC9", "PLOM6"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 15,
        "integration_time": 5.0,
        "priority": 2,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": "datetime.datetime(2025, 4, 22, 6, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 4, 22, 7, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 4, 22, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 252: DataEnrichmentObjective
    (
        "Execute comprehensive DataEnrichmentObjective for collection against targets 27635, 51982, 94063, and 82105. Deploy the following sensor suite: SBNY5, DHBI8, and KYLN3. Implement with S//FOUO classification level operating in REAL mode with SIDEREAL collection technique. Configure system to observe maximum of 10 RSOs with 15 revisits per hour. Operation begins at 2025-06-11 18:40:00+00:00 and runs until 2025-06-12 06:30:00+00:00. Set mission priority to 15 and configure binning to level 2. Disable visibility check for continuous collection regardless of geometric positioning."
    ): {
        "classification_marking": "S//FOUO",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["27635", "51982", "94063", "82105"],
        "sensor_name_list": ["SBNY5", "DHBI8", "KYLN3"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "binning": 2,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 6, 11, 18, 40, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 12, 6, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 253: SensorCheckoutObjective
    (
        "Conduct urgent SensorCheckoutObjective for newly deployed asset NHWK2 following recent calibration. Utilize TS//FOUO classification with operational parameters targeting LEO regime. Execute with SIMULATED data mode and RATE_TRACK collection methodology. System checkout requires 7 revisits per hour with 8 frames per observation and 2.5 second integration time. Implement binning level 1 and priority level 4. Schedule checkout window from 2025-07-01 02:30:00+00:00 to 2025-07-01 09:45:00+00:00 with patience threshold of 20 minutes. Intent submission deadline is 2025-07-01 01:30:00+00:00."
    ): {
        "classification_marking": "TS//FOUO",
        "sensor_name": "NHWK2",
        "orbital_regime": "LEO",
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 7.0,
        "number_of_frames": 8,
        "integration_time": 2.5,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 7, 1, 2, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 7, 1, 9, 45, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": "datetime.datetime(2025, 7, 1, 1, 30, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 254: BaselineAutonomyObjective
    (
        "Establish standing BaselineAutonomyObjective with UUID 'a1b2c3d4-e5f6-7890-abcd-1234567890ef' for continuous network monitoring. Apply S classification level with EXERCISE mode and set priority to 850. Monitor catalog IDs 24972, 39651, and 88435 indefinitely, with no planned termination date. Correlate with RSO IDs 45627, 77219, 33095, 68441, and 92087 for enhanced tracking confidence. Set frame type to LIGHT for maximum detection sensitivity."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-abcd-1234567890ef",
        "classification_marking": "S",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "priority": 850,
        "baseline_autonomy_rso": "24972,39651,88435",
        "objective_end_time": None,
        "rso_id_list": ["45627", "77219", "33095", "68441", "92087"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 255: CatalogMaintenanceObjective
    (
        "Alright, let's get this done. We need to create a CatalogMaintenanceObjective, a real important one. "
        "This here objective is for keeping our eyes on the sky, specifically for sensors 'BIGTEX01' and 'LSTAR02'. "
        "Mark it as 'U//FOUO', and we're gonna run this in 'REAL' mode, 'cause this ain't no drill. "
        "Set the priority to a nice round 50. We need some patience on this one, let's say 45 minutes, and an end time offset of 60 minutes. "
        "Make darn sure that visibility check is set to true. We ain't lookin' for ghosts. "
        "This whole shebang needs to kick off on 2025-06-15 at 10:00:00+00:00 and wrap it up by 2025-06-15 18:00:00+00:00. "
        "We'll be using 'RATE_TRACK' for our tracking type, focusing on the 'MEO' orbital regime. "
        "And listen up, the RSO IDs we're interested in are '55001', '55002', and '55003'. Don't forget the binning, set it to 2."
    ): {
        "binning": 2,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "end_time_offset_minutes": 60,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 50,
        "rso_id_list": ["55001", "55002", "55003"],
        "sensor_name_list": ["BIGTEX01", "LSTAR02"],
        "visibility_check": True,
    },
    # New Example 256: SearchObjective
    (
        "Saddle up! We're creating a SearchObjective. We're lookin' for target 'TGT-ALPHA-789' using the 'HAWKEYE05' sensor. "
        "This needs a 'C' classification marking, and we'll run it in 'SIMULATED' mode for practice. Priority is critical, set it to 3. "
        "Tracking type will be 'SIDEREAL'. This search is planned to start on 2025-07-01 at 02:00:00+00:00 and should conclude by 2025-07-01 05:30:00+00:00. "
        "Our initial offset needs to be 120 seconds, with a final offset of 150 seconds. Let's get a frame overlap of 60 percent, that's 0.6. "
        "The end time offset should be 30 minutes. We'll use a 'CROSS_TRACK' search type, and the search itself should commence 10 minutes after the objective starts. "
        "Set the number of frames to 10 and integration time to 0.5 seconds."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "final_offset": 150,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 120,
        "integration_time": 0.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 7, 1, 5, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 1, 2, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 7, 1, 2, 10, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 10 minutes
        "search_type": "CROSS_TRACK",
        "sensor_name": "HAWKEYE05",
        "target_id": "TGT-ALPHA-789",
        "visibility_check": False,
    },
    # New Example 257: GeodssRevisitObjective
    (
        "Alright, cowboys and cowgirls, listen close! We're setting up a GeodssRevisitObjective. This is for targets 'SATCOM-X1', 'NAVSTAR-Y2' using sensors 'GEODSS-MAUI', 'GEODSS-DIEGO'. "
        "Slap a 'TS' marking on this, 'REAL' data mode, you hear? Priority is a solid 7. We'll use the default 'RATE_TRACK_SIDEREAL'. "
        "It's gonna start on 2025-08-10 at 20:00:00+00:00. Now for the nitty-gritty: readout_rate is 0 (that's 1MHz), gain_setting is 1 (Low Gain, mind you), "
        "soi_filter is 2 (10% Light), auto_track_type is 2 (Manual), camera_mode is 1 (Zoomed EBS), array_kind is 0 (Main), "
        "binning_mode is 0 (No Binning), and scan_mode will be 0 (Continuous). We'll set patience to 25 minutes and number of frames to 8."
    ): {
        "acquisition_type": 0,  # Default, not specified in prompt but good to have
        "array_kind": 0,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "number_of_frames": 8,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 10, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 25,
        "priority": 7,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEODSS-MAUI", "GEODSS-DIEGO"],
        "soi_filter_position": 2,
        "target_id_list": ["SATCOM-X1", "NAVSTAR-Y2"],
        "visibility_check": False,  # Default
    },
    # New Example 258: PeriodicRevisitObjective
    (
        "Gather 'round, folks! Time for a PeriodicRevisitObjective. We're targeting 'OBJ-DELTA-007' and 'OBJ-GAMMA-008' with sensors 'TRACKMSTR-A' and 'TRACKMSTR-B'. "
        "This one's marked 'U', data mode is 'EXERCISE', and priority is a low-key 15. Let's be patient for 35 minutes. "
        "We want 4 revisits per hour, if you please. Objective starts 2025-09-05 at 12:00:00+00:00 and ends 2025-09-06 12:00:00+00:00. "
        "Don't you go ignoring other objective intent submissions, set that to false. Optimal frames per hour can stay at the default 400. "
        "We'll also need 3 frames per intent with an integration time of 1.5 seconds. Set visibility check to true, we only want to see what's actually there."
    ): {
        "classification_marking": "U",
        "target_id_list": ["OBJ-DELTA-007", "OBJ-GAMMA-008"],
        "sensor_name_list": ["TRACKMSTR-A", "TRACKMSTR-B"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 35,
        "revisits_per_hour": 4.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 6, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 259: UctObservationObjective
    (
        "Listen up, team! We're setting up a UctObservationObjective. We're focusing on UCT RSOs 'UCT-UNKNOWN-01' and 'UCT-MYSTERY-02' using sensors 'SKYWATCH-7' and 'NIGHTOWL-3'. "
        "Marking is 'S', data mode 'REAL', for the 'XGEO' regime. Priority is gonna be 8. We need about 5.5 revisits per hour. "
        "Let's start this on 2025-10-20 at 00:00:00+00:00. No specific end time for this one, let it run its course. "
        "We'll sort by the brightest UCT, that's a true. Set the end time offset to 75 minutes. "
        "Number of frames will be 4, and integration time will be 2.2 seconds. And remember, visibility check should be on."
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["UCT-UNKNOWN-01", "UCT-MYSTERY-02"],
        "sensor_name_list": ["SKYWATCH-7", "NIGHTOWL-3"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 5.5,
        "number_of_frames": 4,
        "integration_time": 2.2,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 20, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 8,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 260: SingleIntentObjective
    (
        "Alright, here's a quick one: a SingleIntentObjective. Target ID is 'TGT-SINGLE-555', and the RSO ID is 'RSO-XYZ-999'. "
        "We'll use sensor 'ONE khắp SHOT-01' for this. Classification is 'U//FOUO', data mode 'TEST'. "
        "Tracking type 'RATE_TRACK'. Set priority to a high 4. "
        "This is scheduled for 2025-11-11 at 11:11:00+00:00. We need exactly 1 frame, with an integration time of 10 seconds. "
        "Set the binning to 4. This objective has no specific end time, just the start."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "TGT-SINGLE-555",
        "rso_id": "RSO-XYZ-999",
        "sensor_name_list": [
            "ONE khắp SHOT-01"
        ],  # Needs to be a list based on schema, even if one
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 1,
        "integration_time": 10.0,
        "priority": 4,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 11, 11, 11, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 261: DataEnrichmentObjective
    (
        "Howdy! Let's get some DataEnrichmentObjective work done. We are looking at targets 'DEBRIS-A1', 'DEBRIS-B2', and 'DEBRIS-C3', using sensors 'CLEANUP-CREW1' and 'ORBITPATROL-5'. "
        "This is 'C' classified, 'REAL' data mode. Let's use 'SIDEREAL' tracking this time. We want to observe a maximum of 5 RSOs. "
        "And we need 8 revisits per hour. Start this operation on 2025-12-01 at 06:00:00+00:00. No explicit end time. "
        "Priority is 25. And make sure that visibility_check is true, very important for this kind of work."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["DEBRIS-A1", "DEBRIS-B2", "DEBRIS-C3"],
        "sensor_name_list": ["CLEANUP-CREW1", "ORBITPATROL-5"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 262: SensorCheckoutObjective
    (
        "Alright, time to give our gear a good ol' SensorCheckoutObjective. This is for sensor 'NEW-SENSOR-9000'. "
        "Classification marking is 'U', orbital regime is 'LEO'. We'll use 'TEST' data_mode. "
        "Tracking's gonna be 'RATE_TRACK'. Set revisits_per_hour to 2.0. "
        "This checkout starts on 2026-01-15 at 09:00:00+00:00, and we'll give it until 2026-01-15 17:00:00+00:00 to finish. "
        "Patience can be the default 30 minutes. Number of frames: 6. Integration time: 3 seconds. Priority: 9."
    ): {
        "classification_marking": "U",
        "sensor_name": "NEW-SENSOR-9000",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 30,
        "revisits_per_hour": 2.0,
        "number_of_frames": 6,
        "integration_time": 3.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 15, 17, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 9,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 263: BaselineAutonomyObjective
    (
        "Here comes the big one, the BaselineAutonomyObjective. This needs a specific UUID, let's use 'fadgumb1-e89b-12d3-a456-texanpride01'. "
        "Markings are 'S', data mode 'REAL', frame type 'DARK' for this one. Priority is super high, set it to 1, the lowest number for highest importance. "
        "The baseline autonomy RSOs we're tracking, using their catalog IDs, are '25544,25545,28654'. "
        "This objective ain't got no end time, it's gonna run till the cows come home. RSO ID list for good measure includes 'SATCAT-A', 'SATCAT-B', though the system will manage 'em."
    ): {
        "objective_uuid": "fadgumb1-e89b-12d3-a456-texanpride01",
        "classification_marking": "S",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 1,  # Highest priority
        "baseline_autonomy_rso": "25544,25545,28654",
        "objective_end_time": None,  # For continuous running
        "rso_id_list": ["SATCAT-A", "SATCAT-B"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # Example 264: CatalogMaintenanceObjective
    (
        "Create a CatalogMaintenanceObjective for sensors TXS01 and AZG03. Use S markings, SIMULATED mode, top priority 1. "
        "Set a patience of 45 minutes and an end time offset of 30 minutes. We don't need a visibility check for this round. "
        "Start this job up at 2025-06-15 04:00:00+00:00 and run it until 2025-06-16 03:59:59+00:00. "
        "Stick with RATE_TRACK tracking in the GEO regime. Make sure you include RSO IDs '98765' and '45678'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",  # Default value
        "objective_end_time": "datetime.datetime(2025, 6, 16, 3, 59, 59, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 4, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 45,
        "priority": 1,
        "rso_id_list": ["98765", "45678"],
        "sensor_name_list": ["TXS01", "AZG03"],
        "visibility_check": False,
    },
    # Example 265: SearchObjective
    (
        "Now listen here, I need a SearchObjective for target 'TARGET-ALPHA-7' using the sensor RME55. "
        "Mark this one C, run it in REAL mode, with a mighty important priority of 3. We'll use SIDEREAL tracking. "
        "Start the whole darn thing at 2025-07-10 14:00:00+00:00 and have it wrapped up by 2025-07-10 16:00:00+00:00. "
        "Let's set the initial offset to 90 seconds and the final offset to 120 seconds. "
        "We want a frame overlap of 80% and an end time offset of 50 minutes. "
        "Use the search type CROSS_TRACK, and the search itself should kick off 20 minutes after the objective begins."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 50,
        "final_offset": 120,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",  # Default value
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 7, 10, 16, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        # Search start time is 20 minutes after objective start (14:00 + 20 mins = 14:20)
        "search_start_time": "datetime.datetime(2025, 7, 10, 14, 20, 0, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "RME55",
        "target_id": "TARGET-ALPHA-7",
        "visibility_check": False,
    },
    # Example 266: GeodssRevisitObjective
    (
        "How about a GeodssRevisitObjective for targets 'TARGET-BETA-1', 'TARGET-BETA-2', and 'TARGET-BETA-3'? "
        "Run this with sensors LMNT88 and LMNT89. Classification is U//FOUO, mode is REAL, and the priority is a respectable 7. "
        "Use RATE_TRACK_SIDEREAL tracking. Get this started at 2025-08-01 20:00:00+00:00. "
        "Set the readout rate to 0 (1MHz), gain setting to 1 (Low Gain), SOI filter to 2 (10% Light), "
        "auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), array kind to 1 (Photometer), "
        "binning mode to 0 (No Binning), and scan mode to 0 (Continuous). Don't worry about ignoring other objective submissions this time."
    ): {
        "acquisition_type": 0,  # Default value
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default value
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default value
        "data_mode": "REAL",
        "frame_type": "LIGHT",  # Default value
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default value
        "num_skip_frames": 0,  # Default value
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 1, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default value
        "optimal_frames_per_hour": 400,  # Default value
        "overscan": 0,  # Default value
        "patience_minutes": 30,  # Default value
        "priority": 7,
        "rate_track_verify": 0,  # Default value
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["LMNT88", "LMNT89"],
        "soi_filter_position": 2,
        "target_id_list": ["TARGET-BETA-1", "TARGET-BETA-2", "TARGET-BETA-3"],
        "visibility_check": False,  # Default value
    },
    # Example 267: PeriodicRevisitObjective
    (
        "Time for a PeriodicRevisitObjective! Let's hit targets 'P-TARGET-4', 'P-TARGET-5' with sensors RME60. "
        "This one needs to be S classified, in EXERCISE mode, and have a priority of 5. "
        "Patience should be 20 minutes, and we *will* ignore other objective submissions for this run. "
        "Start this whole shebang at 2025-09-05 10:00:00+00:00. "
        "We aim for 15 revisits per hour, with 10 frames per intent, and an integration time of 3.5 seconds."
    ): {
        "classification_marking": "S",
        "target_id_list": ["P-TARGET-4", "P-TARGET-5"],
        "sensor_name_list": ["RME60"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default value
        "frame_type": "LIGHT",  # Default value
        "visibility_check": False,  # Default value
        "patience_minutes": 20,
        "revisits_per_hour": 15.0,
        "number_of_frames": 10,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,  # Default value
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # Example 268: UctObservationObjective
    (
        "We've got some unidentified buddies we need to keep an eye on. "
        "Let's set up a UctObservationObjective for UCT RSOs 'UCT-999', 'UCT-888'. "
        "Use sensors TXS05 and AZG07 for this. Classification is TS, mode is REAL, regime is MEO, and priority is 4. "
        "We need 8 revisits per hour. Kick it off at 2025-10-20 22:00:00+00:00. "
        "Make sure to enable sorting by brightest UCT and set the end time offset to 90 minutes. "
        "Visibility check is absolutely necessary. We want 7 frames per intent with an integration time of 1.8 seconds."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["UCT-999", "UCT-888"],
        "sensor_name_list": ["TXS05", "AZG07"],
        "data_mode": "REAL",  # Default value
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default value
        "frame_type": "LIGHT",  # Default value
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default value
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 1.8,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 20, 22, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 4,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # Example 269: SingleIntentObjective
    (
        "Create a SingleIntentObjective for target ID 'ONE-SHOT-TARGET' and RSO ID 'ONE-SHOT-RSO' using sensor RME70. "
        "Classification is U, mode is TEST, tracking is RATE_TRACK. Set the priority to 8. "
        "Start the objective at 2025-11-11 07:00:00+00:00. "
        "We need 3 frames with an integration time of 5 seconds and binning set to 1."
    ): {
        "classification_marking": "U",
        "target_id": "ONE-SHOT-TARGET",
        "rso_id": "ONE-SHOT-RSO",
        "sensor_name_list": ["RME70"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default value
        "number_of_frames": 3,
        "integration_time": 5.0,
        "priority": 8,
        "binning": 1,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 11, 7, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # Example 270: DataEnrichmentObjective
    (
        "Alright now, let's get some DataEnrichmentObjective going for targets 'DATA-TARGET-A', 'DATA-TARGET-B', 'DATA-TARGET-C', 'DATA-TARGET-D'. "
        "Use sensors LMNT90 and LMNT91. This objective is C classified, runs in REAL mode, and uses RATE_TRACK_SIDEREAL tracking. "
        "Priority is set at 15. We wanna observe a maximum of 10 RSOs and get 14 revisits per hour. "
        "Start this job at 2025-12-01 15:00:00+00:00. Visibility check is true for this operation."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",  # Default value
        "objective_uuid": None,
        "target_id_list": [
            "DATA-TARGET-A",
            "DATA-TARGET-B",
            "DATA-TARGET-C",
            "DATA-TARGET-D",
        ],
        "sensor_name_list": ["LMNT90", "LMNT91"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default value
        "frame_type": "LIGHT",  # Default value
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 14.0,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 15, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # Example 271: SensorCheckoutObjective
    (
        "Let's run a check on sensor AZG10. Create a SensorCheckoutObjective for it. "
        "Classification is U//FOUO, mode is REAL, and we'll focus on the XGEO regime. Use SIDEREAL tracking. "
        "Priority is 6. We need 2 revisits per hour for this checkout. Start it up at 2026-01-15 09:00:00+00:00. "
        "Visibility check is a must. We need 12 frames with an integration time of 1 second."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "AZG10",
        "orbital_regime": "XGEO",
        "data_mode": "REAL",  # Default value
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default value
        "visibility_check": True,
        "patience_minutes": 30,  # Default value
        "revisits_per_hour": 2.0,
        "number_of_frames": 12,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 6,
        "objective_name": "SensorCheckoutObjective",
    },
    # Example 272: BaselineAutonomyObjective
    (
        "Get a BaselineAutonomyObjective cooked up with the UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. "
        "Classify it S, use REAL mode, DARK frame type, and set the priority to a high 500. "
        "For the baseline, track catalog IDs '23456' and '78901'. "
        "Include RSO IDs 'RSO-BBB', 'RSO-CCC', and 'RSO-DDD' as well. Let this run continuously, so no end time needed."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "S",
        "data_mode": "REAL",  # Default value
        "frame_type": "DARK",
        "priority": 500,
        "baseline_autonomy_rso": "23456,78901",
        "objective_end_time": None,
        "rso_id_list": ["RSO-BBB", "RSO-CCC", "RSO-DDD"],
        "objective_name": "BaselineAutonomyObjective",  # Default value
    },
    # New Example 273: CatalogMaintenanceObjective
    (
        "Well shoot, partner! I need ya to whip up a CatalogMaintenanceObjective for them sensors RME09 and LMNT03. "
        "Mark it with TS classification and set it to REAL mode, ya hear? Give it a priority of 3 - that's mighty important! "
        "Set the patience to 15 minutes 'cause we ain't in no hurry, and make the end time offset 30 minutes. "
        "We're trackin' objects in the GEO regime with RATE_TRACK_SIDEREAL. Start at 2025-06-15 03:45:00+00:00 and end at 2025-06-15 07:30:00+00:00. "
        "And don't forget to check visibility - set that to true! RSO ID list includes '33445,78901,12345'."
    ): {
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 7, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 3, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 15,
        "priority": 3,
        "rso_id_list": ["33445", "78901", "12345"],
        "sensor_name_list": ["RME09", "LMNT03"],
        "visibility_check": True,
    },
    # New Example 274: SearchObjective
    (
        "Howdy there! Need ya to set up a SearchObjective for target 54321 usin' that fancy sensor BRZ08. "
        "Slap a C marking on it and run it in REAL mode with priority 4. We're gonna use RATE_TRACK trackin' and CROSS_TRACK search type. "
        "Fire it up at 2025-08-02 16:30:00+00:00 and let 'er rip until 2025-08-02 19:45:00+00:00. "
        "Set the initial offset to 45 seconds and final offset to 75 seconds with 60% frame overlap. "
        "Make the end time offset 35 minutes and no visibility check needed. The search should start 20 minutes after objective start."
    ): {
        "classification_marking": "C",
        "target_id": "54321",
        "sensor_name": "BRZ08",
        "search_type": "CROSS_TRACK",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 45,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 35,
        "priority": 4,
        "objective_start_time": "datetime.datetime(2025, 8, 2, 16, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 2, 19, 45, tzinfo=TzInfo(UTC))",
        "search_start_time": "datetime.datetime(2025, 8, 2, 16, 50, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 275: GeodssRevisitObjective
    (
        "Listen up, pardner! I'm needin' a GeodssRevisitObjective for targets 23456,78901 with them sensors RME25,LMNT28. "
        "Make it TS//FOUO classification in REAL mode with RATE_TRACK_SIDEREAL trackin' and priority 8. "
        "Start this rodeo at 2025-07-22 13:15:00+00:00. For them fancy settings, set readout_rate 0 (1MHz), gain_setting 1 (Low Gain), "
        "soi_filter 2 (10% Light), auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 0 (Main), "
        "binning_mode 0 (No Binning), and scan_mode 0 (Continuous). It's like ridin' a mechanical bull!"
    ): {
        "classification_marking": "TS//FOUO",
        "target_id_list": ["23456", "78901"],
        "sensor_name_list": ["RME25", "LMNT28"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "priority": 8,
        "objective_start_time": "datetime.datetime(2025, 7, 22, 13, 15, tzinfo=TzInfo(UTC))",
        "readout_rate_setting": 0,
        "gain_setting": 1,
        "soi_filter_position": 2,
        "auto_track_type": 2,
        "camera_mode": 1,
        "array_kind": 0,
        "binning_mode": 0,
        "scan_mode": 0,
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 276: PeriodicRevisitObjective
    (
        "Y'all listen now! I need a PeriodicRevisitObjective for them targets 98765,54321,13579 usin' sensors RME11,LMNT13. "
        "Make it U//FOUO classification, REAL mode, and set priority to 7. We need patience of 45 minutes and 8 revisits per hour - "
        "keepin' a close eye on them targets like a hawk on a field mouse! Set number of frames to 7 and integration time to 3.5 seconds. "
        "Objective starts at 2025-09-05 22:30:00+00:00. And don't you forget to set optimal frames per hour to 350! Visibility check needs to be true, ya hear?"
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["98765", "54321", "13579"],
        "sensor_name_list": ["RME11", "LMNT13"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 45,
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 3.5,
        "priority": 7,
        "optimal_frames_per_hour": 350,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 277: UctObservationObjective
    (
        "Well butter my biscuit! I need a UctObservationObjective for them UCT RSOs 45678,98765,23456 usin' sensors RME28,LMNT31. "
        "Tag it with S classification in REAL mode, MEO regime, and priority 6. We need 4 revisits per hour and 40 minutes end time offset. "
        "Start at 2025-10-18 05:15:00+00:00. Set frames to 6 and integration time to 2.5 seconds. "
        "And make sure to sort by brightest UCT - we want the shiniest objects first, like pickin' the biggest apple from the tree!"
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["45678", "98765", "23456"],
        "sensor_name_list": ["RME28", "LMNT31"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "revisits_per_hour": 4.0,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "end_time_offset_minutes": 40,
        "objective_start_time": "datetime.datetime(2025, 10, 18, 5, 15, tzinfo=TzInfo(UTC))",
        "priority": 6,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 278: SingleIntentObjective
    (
        "Alrighty then! Gotta create a SingleIntentObjective with target ID 87654 and RSO ID 34567. "
        "We'll be usin' sensors RME37,LMNT39. Make it C//FOUO classification in REAL mode with SIDEREAL trackin'. "
        "Priority's 9, so it's pretty darn important! Set number of frames to 8 and integration time to 1.5 seconds. "
        "Use binning 4 for this one. Start the objective at 2025-11-03 08:10:00+00:00. Quick and easy like ropin' a calf at the county fair!"
    ): {
        "classification_marking": "C//FOUO",
        "target_id": "87654",
        "rso_id": "34567",
        "sensor_name_list": ["RME37", "LMNT39"],
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 8,
        "integration_time": 1.5,
        "priority": 9,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 11, 3, 8, 10, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 279: DataEnrichmentObjective
    (
        "Listen up, cowpokes! Need a DataEnrichmentObjective for targets 11223,33445,55667 with sensors RME42,LMNT44. "
        "Make it U classification, TEST mode, RATE_TRACK trackin', and set max RSO to observe as 10. "
        "We need 15 revisits per hour - more frequent than a rooster crowin' at dawn! Objective starts at 2025-12-15 17:45:00+00:00. "
        "Set priority to 15 and make sure visibility check is true. This data's gonna be richer than Texas oil fields!"
    ): {
        "classification_marking": "U",
        "data_mode": "TEST",
        "target_id_list": ["11223", "33445", "55667"],
        "sensor_name_list": ["RME42", "LMNT44"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 12, 15, 17, 45, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 280: SensorCheckoutObjective
    (
        "Hey there, buckaroo! Need ya to rustle up a SensorCheckoutObjective with sensor RME50. "
        "Mark it S//FOUO classification in REAL mode with RATE_TRACK_SIDEREAL trackin' in the MEO regime. "
        "Priority's 5, and we need 2 revisits per hour. Set patience to 40 minutes and make sure visibility check is true. "
        "Number of frames is 4 with integration time of 3 seconds. Start at 2026-01-10 12:30:00+00:00. "
        "This checkout's gonna be smoother than a tumbleweed rollin' across the prairie!"
    ): {
        "classification_marking": "S//FOUO",
        "sensor_name": "RME50",
        "orbital_regime": "MEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.0,
        "number_of_frames": 4,
        "integration_time": 3,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 12, 30, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 281: BaselineAutonomyObjective
    (
        "Yeehaw, partner! I need a BaselineAutonomyObjective with UUID '789e4567-e89b-12d3-a456-426614174999'. "
        "Make it C classification, TEST mode, LIGHT frame type, and priority 950. "
        "Use the followin' catalog IDs: 18190,19220,20230 for baseline autonomy RSOs. "
        "Add RSO IDs 22334,44556,66778 to the list. This here objective ain't got no end time 'cause it should run continuous like a river flowin' through the canyon!"
    ): {
        "objective_uuid": "789e4567-e89b-12d3-a456-426614174999",
        "classification_marking": "C",
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "priority": 950,
        "baseline_autonomy_rso": "18190,19220,20230",
        "objective_end_time": None,
        "rso_id_list": ["22334", "44556", "66778"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 282: CatalogMaintenanceObjective
    (
        "Establish a CatalogMaintenanceObjective for sensors AUS09 and BRL33 utilizing S classification markings. Configure for REAL mode operation with elevated priority of 8 and patience threshold of 15 minutes. Implement end time offset parameter of 30 minutes with orbital regime set to MEO. Ensure visibility check is enabled. Schedule commencement at 2025-06-15T08:45:00+00:00 with termination at 2025-06-15T14:30:00+00:00. Employ RATE_TRACK_SIDEREAL tracking methodology. Target RSO identification numbers include '33456', '78901', and '12567'."
    ): {
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 8, 45, tzinfo=TzInfo(UTC))",
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 8,
        "rso_id_list": ["33456", "78901", "12567"],
        "sensor_name_list": ["AUS09", "BRL33"],
        "visibility_check": True,
    },
    # New Example 283: SearchObjective
    (
        "Initiate a SearchObjective for target 98765 with UKR45 sensor. Apply C classification marking and SIMULATED data mode. Assign priority level 3 with RATE_TRACK tracking protocol. Commence operations on 2025-08-10T13:15:00+00:00 and conclude by 2025-08-10T17:45:00+00:00. Establish initial offset of 45 seconds, final offset of 60 seconds, and frame overlap of 65%. Set end time offset to 50 minutes. Configure for ALONG_TRACK search pattern with search initiation 20 minutes following objective start time."
    ): {
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 50,
        "final_offset": 60,
        "frame_overlap_percentage": 0.65,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 8, 10, 17, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 10, 13, 15, tzinfo=TzInfo(UTC))",
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 8, 10, 13, 35, tzinfo=TzInfo(UTC))",
        "search_type": "ALONG_TRACK",
        "sensor_name": "UKR45",
        "target_id": "98765",
        "visibility_check": False,
    },
    # New Example 284: GeodssRevisitObjective
    (
        "Construct a GeodssRevisitObjective targeting objects 45678 and 91234 via sensors PAC10 and EUR22. Implementation should adhere to TS classification protocols and operate in TEST mode with SIDEREAL tracking methodology. Configure for priority level 7. Begin execution on 2025-09-05T22:10:00+00:00. Set readout rate to 0 (1MHz), gain setting to 1 (Low Gain), soi filter to 2 (10% Light), auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), array kind to 1 (Photometer), binning mode to 0 (No Binning), and scan mode to 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "command": 0,
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "num_observations": 1,
        "num_skip_frames": 0,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 10, tzinfo=TzInfo(UTC))",
        "observation_interval": 0,
        "patience_minutes": 30,
        "priority": 7,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "scan_mode": 0,
        "sensor_name_list": ["PAC10", "EUR22"],
        "soi_filter_position": 2,
        "target_id_list": ["45678", "91234"],
        "visibility_check": False,
    },
    # New Example 285: PeriodicRevisitObjective
    (
        "Define a PeriodicRevisitObjective for targets 56789 and 34567 utilizing the NOR11 and FIN05 sensors. Apply U//FOUO classification restrictions and configure for EXERCISE mode execution. Maintain RATE_TRACK collection methodology with priority rating of 4. Set patience parameter to 40 minutes. Configure for 3 revisits per hour with 4 frames per observation and 3.5 second integration time. Initiate objective at 2025-07-15T04:30:00+00:00. Enable visibility verification and optimize for 300 frames per operational hour."
    ): {
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "ignore_other_objective_intent_submissions": False,
        "integration_time": 3.5,
        "number_of_frames": 4,
        "objective_name": "PeriodicRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 15, 4, 30, tzinfo=TzInfo(UTC))",
        "optimal_frames_per_hour": 300,
        "patience_minutes": 40,
        "priority": 4,
        "revisits_per_hour": 3.0,
        "sensor_name_list": ["NOR11", "FIN05"],
        "target_id_list": ["56789", "34567"],
        "visibility_check": True,
    },
    # New Example 286: UctObservationObjective
    (
        "Implement a UctObservationObjective for UCT RSOs 23456, 78945, and 10111 employing AFR17 and AFR22 sensors. Designate classification as S and operate in REAL data mode with SIDEREAL tracking in the GEO regime. Set priority to 9 with 8 revisits hourly. Begin at 2025-11-03T15:20:00+00:00 with operational termination scheduled for 2025-11-03T23:45:00+00:00. Configure with 40 minutes end time offset and enable both visibility validation and brightness-based UCT sorting. Specify 8 frames per observation with 5 second integration periods."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 40,
        "frame_type": "LIGHT",
        "integration_time": 5,
        "number_of_frames": 8,
        "objective_end_time": "datetime.datetime(2025, 11, 3, 23, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "UctObservationObjective",
        "objective_start_time": "datetime.datetime(2025, 11, 3, 15, 20, tzinfo=TzInfo(UTC))",
        "orbital_regime": "GEO",
        "patience_minutes": 30,
        "priority": 9,
        "revisits_per_hour": 8.0,
        "sensor_name_list": ["AFR17", "AFR22"],
        "sort_by_brightest_uct": True,
        "uct_rso_id_list": ["23456", "78945", "10111"],
        "visibility_check": True,
    },
    # New Example 287: SingleIntentObjective
    (
        "Formulate a SingleIntentObjective with target identifier 45678 and RSO identifier 98712 utilizing JPN33 and KOR19 sensors. Employ C classification protocol and SIMULATED operational mode with RATE_TRACK tracking methodology at priority level 6. Schedule commencement for 2025-12-10T09:45:00+00:00 with conclusion at 2025-12-10T14:30:00+00:00. Specify 10 frames per observation with 1.5 second integration duration and binning factor of 4."
    ): {
        "binning": 4,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",
        "integration_time": 1.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 12, 10, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
        "objective_start_time": "datetime.datetime(2025, 12, 10, 9, 45, tzinfo=TzInfo(UTC))",
        "priority": 6,
        "rso_id": "98712",
        "sensor_name_list": ["JPN33", "KOR19"],
        "target_id": "45678",
    },
    # New Example 288: DataEnrichmentObjective
    (
        "Establish a DataEnrichmentObjective for observational targets 11223, 44556, and 77889 employing CAN08 and USA15 sensors. Implement with TS classification restrictions in REAL operational mode. Override default tracking methodology to utilize RATE_TRACK. Configure to monitor maximum of 10 RSOs with 15 revisits per hour. Schedule operational initiation at 2026-01-05T18:30:00+00:00 with termination at 2026-01-06T06:15:00+00:00. Ensure visibility validation is performed prior to intent generation."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "objective_end_time": "datetime.datetime(2026, 1, 6, 6, 15, tzinfo=TzInfo(UTC))",
        "objective_name": "DataEnrichmentObjective",
        "objective_start_time": "datetime.datetime(2026, 1, 5, 18, 30, tzinfo=TzInfo(UTC))",
        "priority": 20,
        "revisits_per_hour": 15.0,
        "sensor_name_list": ["CAN08", "USA15"],
        "target_id_list": ["11223", "44556", "77889"],
        "visibility_check": True,
    },
    # New Example 289: SensorCheckoutObjective
    (
        "Configure a SensorCheckoutObjective with classification designation U//FOUO for sensor AUS21. Establish operational parameters in MEO orbital regime utilizing EXERCISE data mode and SIDEREAL tracking methodology. Assign priority 5 with 2.5 revisits hourly. Begin operations at 2025-10-15T12:00:00+00:00. Deactivate visibility verification and adjust patience threshold to 45 minutes. Specify 6 frames per observation and 2.5 second integration periods with binning factor of 2."
    ): {
        "binning": 2,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",
        "integration_time": 2.5,
        "number_of_frames": 6,
        "objective_name": "SensorCheckoutObjective",
        "objective_start_time": "datetime.datetime(2025, 10, 15, 12, 0, tzinfo=TzInfo(UTC))",
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 5,
        "revisits_per_hour": 2.5,
        "sensor_name": "AUS21",
        "visibility_check": False,
    },
    # New Example 290: BaselineAutonomyObjective
    (
        "Instantiate a BaselineAutonomyObjective with UUID '987f6543-a21b-52c7-d890-123456789abc' utilizing S classification protocol and TEST mode operation with LIGHT frame acquisition. Implement at priority level 800. Associate with RSO identifiers 44455, 66677, and 88899 alongside catalog designations 12340 and 56780 to ensure continuous monitoring without specified end time."
    ): {
        "objective_uuid": "987f6543-a21b-52c7-d890-123456789abc",
        "classification_marking": "S",
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "priority": 800,
        "baseline_autonomy_rso": "12340,56780",
        "objective_end_time": None,
        "rso_id_list": ["44455", "66677", "88899"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 291: CatalogMaintenanceObjective
    (
        "Kindly formulate a CatalogMaintenanceObjective. This particular objective is designated for sensors 'RDLF3' and 'GRNT77', "
        "which are to operate under 'C' classification markings. The operational mode is specified as 'SIMULATED'. "
        "We shall employ 'SIDEREAL' tracking, specifically targeting assets within the MEO orbital regime. "
        "A patience duration of 45 minutes is requested by the operators, and this is to be accompanied by an end time offset of precisely 30 minutes. "
        "The scheduling priority for this crucial task is set at 25. "
        "Furthermore, the RSO ID list for this objective encompasses '98765' and '54321'. "
        "The objective is meticulously scheduled to commence at 2025-11-10 10:00:00+00:00, UTC, and it is planned to conclude at 2025-11-10 15:30:00+00:00, UTC. "
        "It is imperative that the visibility check is activated, and please also specify a binning value of 2 for the imaging process."
    ): {
        "classification_marking": "C",
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "end_time_offset_minutes": 30,
        "priority": 25,
        "sensor_name_list": ["RDLF3", "GRNT77"],
        "rso_id_list": ["98765", "54321"],
        "objective_start_time": "datetime.datetime(2025, 11, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 10, 15, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": 2,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 292: SearchObjective
    (
        "Let us construct a SearchObjective for the elusive target identifier 'targetAlpha001'. "
        "This important search operation will be executed by the capable sensor 'XIP99'. "
        "The chosen search methodology, after careful consideration, will be 'CROSS_TRACK'. "
        "This operation falls under the 'TS' classification marking, a high level of sensitivity, and will run in 'EXERCISE' mode to test our capabilities. "
        "The tracking type shall be 'RATE_TRACK', which is distinct from other tracking modes. "
        "Please set the initial offset to 120 seconds and the final offset to 180 seconds; these parameters are critical for defining the search window. "
        "A frame overlap of 60% is required to ensure comprehensive coverage. "
        "The end time offset for scheduling purposes is to be 50 minutes, with a priority of 3, reflecting its importance. "
        "The objective is slated to begin at 2025-12-01 08:00:00+00:00 UTC and will conclude at 2025-12-01 12:00:00+00:00 UTC. "
        "The actual search pattern should commence 20 minutes subsequent to the objective's formal initiation. "
        "We also require the collection of precisely 10 frames, each with an integration time of 0.5 seconds."
    ): {
        "classification_marking": "TS",
        "target_id": "targetAlpha001",
        "sensor_name": "XIP99",
        "search_type": "CROSS_TRACK",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "initial_offset": 120,
        "final_offset": 180,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 50,
        "priority": 3,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 12, 1, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 0.5,
        "search_start_time": "datetime.datetime(2025, 12, 1, 8, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
    },
    # New Example 293: GeodssRevisitObjective
    (
        "We shall now meticulously define a GeodssRevisitObjective. This particular objective is targeted towards 'geoTarget7' and 'geoTarget8', "
        "making use of the specialised sensors 'GEO01' and 'GEO02'. "
        "The classification marking for all data produced will be 'U//FOUO', and the operational data mode is firmly set to 'REAL'. "
        "We require the 'RATE_TRACK_SIDEREAL' tracking capability for this task. "
        "The objective itself is scheduled to formally commence on the 15th of January, 2026, precisely at 03:00:00 Coordinated Universal Time. "
        "Several key GEODSS-specific parameters must be configured: the readout_rate setting shall be 0 (which corresponds to 1MHz), "
        "the gain_setting is to be 1 (indicative of Low Gain), and the soi_filter position should be 2 (allowing 10% Light). "
        "Furthermore, the auto_track_type is designated as 2 (Manual operation), the camera_mode will be 1 (Zoomed EBS configuration), "
        "the array_kind selected is 1 (Photometer), with a binning_mode of 0 (No Binning applied), and lastly, the scan_mode is set to 0 (Continuous data acquisition). "
        "A priority level of 7 has been assigned. The number of observations per target pass shall be 3, with a defined observation interval of 120.5 seconds. "
        "Patience for intent completion is established at 25 minutes. For this objective, kindly set `ignore_other_objective_intent_submissions` to `False` and `revisits_per_hour` to 2.0."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["geoTarget7", "geoTarget8"],
        "sensor_name_list": ["GEO01", "GEO02"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 2.0,
        "number_of_frames": None,
        "integration_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 3, 0, 0, tzinfo=TzInfo(UTC))",
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
        "observation_interval": 120.5,
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
    # New Example 294: PeriodicRevisitObjective
    (
        "Please generate a PeriodicRevisitObjective, if you would be so kind. This is specifically for targets identified as 'periodicTgt003' and 'periodicTgt004', "
        "and it will employ the sensor assets 'SNS_A' and 'SNS_B'. The data marking for this operation is 'S'. It shall operate in 'TEST' mode, which is suitable for system evaluations. "
        "A scheduling priority of 4 is assigned to this task, and a patience window of 35 minutes is set for individual intent completions. "
        "It is quite important that you explicitly set the 'ignore other objective submissions' parameter to true for this instance. "
        "The objective itself is scheduled to begin at 2026-02-20 14:00:00+00:00 UTC. Our operational requirements stipulate precisely 3 revisits per hour for each target. "
        "During each revisit, we aim to capture 7 frames, with each frame having an integration time of 1.5 seconds. "
        "The visibility check should be configured to false, and the specified tracking type is RATE_TRACK. "
        "Furthermore, let the overarching objective conclude its activities on 2026-02-22 14:00:00+00:00 UTC. "
        "The intents generated by this objective should have their earliest start time (intent_start_time) as 2026-02-20 14:30:00+00:00 UTC and their latest end time (intent_end_time) as 2026-02-21 14:30:00+00:00 UTC."
    ): {
        "classification_marking": "S",
        "target_id_list": ["periodicTgt003", "periodicTgt004"],
        "sensor_name_list": ["SNS_A", "SNS_B"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 35,
        "revisits_per_hour": 3.0,
        "number_of_frames": 7,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 2, 22, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,
        "objective_uuid": None,
        "intent_start_time": "datetime.datetime(2026, 2, 20, 14, 30, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 2, 21, 14, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 295: UctObservationObjective
    (
        "I should be grateful if you could formulate a UctObservationObjective. This is for Uncorrelated Tracks, specifically RSOs 'uctRSO_X1' and 'uctRSO_Y2'. "
        "The observation will utilize sensors 'UCT_Sensor_Alpha' and 'UCT_Sensor_Beta'. "
        "This objective is to carry a 'U//FOUO' classification marking and will operate in 'REAL' mode. It is particularly designed for the 'XGEO' orbital regime, a challenging environment. "
        "Please assign a priority of 15. The objective is scheduled to begin its operations at 2026-03-10 05:00:00+00:00 UTC. "
        "A critical requirement for this objective is to enable sorting by the brightest UCT, so please ensure that parameter is set accordingly. "
        "The end time offset for scheduling individual intents should be set to 75 minutes, and it is imperative that the visibility check is enabled. "
        "We are aiming for an average of 4.5 revisits per hour for these UCTs, capturing 6 frames per observation, with each frame benefiting from a 2.5 seconds integration time. "
        "For image processing, binning should be set to a value of 4. The overall objective should conclude by 2026-03-11 05:00:00+00:00 UTC."
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["uctRSO_X1", "uctRSO_Y2"],
        "sensor_name_list": ["UCT_Sensor_Alpha", "UCT_Sensor_Beta"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "XGEO",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 4.5,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "binning": 4,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 10, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 3, 11, 5, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 296: SingleIntentObjective
    (
        "Would you be so good as to construct a SingleIntentObjective? This specific objective has a target identifier of 'singleTgt555' and a corresponding RSO identifier of 'singleRSO999'. "
        "It is planned that this objective will utilise the sensor resources 'SensorM' and 'SensorN'. "
        "The classification marking for any resultant data is 'U', with the operational mode set to 'SIMULATED'. For tracking purposes, please employ 'SIDEREAL' tracking. "
        "A priority level of 8 is requested for this particular intent. The objective itself should commence its active period at 2026-04-05 18:30:00+00:00 UTC. "
        "Within this objective, please configure the single intent for the collection of 3 frames, an integration time of 3 seconds per frame, and a camera binning setting of 1. "
        "The intent generated by this objective should be precisely scheduled to start at 2026-04-05 18:35:00+00:00 UTC and must end by 2026-04-05 19:00:00+00:00 UTC. "
        "The overall SingleIntentObjective will conclude its activities at 2026-04-05 20:00:00+00:00 UTC."
    ): {
        "classification_marking": "U",
        "target_id": "singleTgt555",
        "rso_id": "singleRSO999",
        "sensor_name_list": ["SensorM", "SensorN"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 3,
        "integration_time": 3.0,
        "priority": 8,
        "binning": 1,
        "intent_start_time": "datetime.datetime(2026, 4, 5, 18, 35, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 4, 5, 19, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 5, 18, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 4, 5, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 297: DataEnrichmentObjective
    (
        "Let us now turn our attention to creating a DataEnrichmentObjective. This objective is specifically designed for the target identifiers 'enrichTgt01', 'enrichTgt02', and the rather important 'enrichTgt03'. "
        "It will make use of sensor assets 'ENR_S1' and 'ENR_S2'. The data produced will carry a 'C' classification marking and the objective will operate in 'REAL' mode, handling live data. "
        "The designated tracking type for this operation is 'RATE_TRACK'. A key parameter, the maximum number of RSOs to observe concurrently, is set to 7. "
        "Furthermore, there is a requirement for achieving 15 revisits per hour for the specified targets. "
        "The objective is scheduled to initiate its activities on 2026-05-12 at precisely 07:00:00+00:00 UTC. "
        "A priority of 22 has been assigned to this objective, reflecting its significance, and it is absolutely essential that the visibility check is set to true. "
        "The unique objective UUID for tracking and reference shall be 'de-obj-uuid-001'. "
        "For this particular objective, the frame type will be 'DARK', which is a deviation from the usual 'LIGHT' frames. "
        "The objective is planned to run for a duration of 24 hours from its start time."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": "de-obj-uuid-001",
        "target_id_list": ["enrichTgt01", "enrichTgt02", "enrichTgt03"],
        "sensor_name_list": ["ENR_S1", "ENR_S2"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "DARK",
        "binning": None,
        "max_rso_to_observe": 7,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 5, 12, 7, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 5, 13, 7, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 22,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 298: SensorCheckoutObjective
    (
        "We now require the formulation of a SensorCheckoutObjective. The classification marking for any data generated under this objective will be 'TS'. "
        "The sensor under evaluation is 'CHKOUT_SENSOR_7'. This checkout procedure will operate in 'EXERCISE' mode, and it specifically targets the 'LEO' orbital regime for its calibration objects. "
        "'RATE_TRACK_SIDEREAL' has been selected as the collection request type. A priority of 5 is assigned. "
        "The objective is scheduled to commence at 2026-06-30 11:00:00+00:00 UTC. It is crucial for this checkout that the visibility check is enabled (set to true). "
        "A patience window of 40 minutes is allocated for the completion of individual intents. The checkout sequence requires 2.0 revisits per hour, during which it will collect 4 frames per intent. "
        "Each of these frames should have an integration time of 2.2 seconds. The camera binning value for this sensor checkout is 3. "
        "The entire SensorCheckoutObjective is scheduled to conclude its operations exactly 24 hours after it begins, which would be on 2026-07-01 at 11:00:00+00:00 UTC. "
        "The intent start time should be set 5 minutes after the objective start time, and the intent end time should be set to objective end time minus 1 hour."
    ): {
        "classification_marking": "TS",
        "sensor_name": "CHKOUT_SENSOR_7",
        "orbital_regime": "LEO",
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.0,
        "number_of_frames": 4,
        "integration_time": 2.2,
        "binning": 3,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 6, 30, 11, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 7, 1, 11, 0, 0, tzinfo=TzInfo(UTC))",
        "intent_start_time": "datetime.datetime(2026, 6, 30, 11, 5, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2026, 7, 1, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 299: BaselineAutonomyObjective
    (
        "Finally, your esteemed assistance is requested to construct a BaselineAutonomyObjective. "
        "This foundational objective will be uniquely identified by the UUID 'baseline-auto-uuid-987'. "
        "It is to operate with 'U//FOUO' classification markings and will function in 'REAL' data mode, processing live information. "
        "The frame type for imagery collected under this objective is 'LIGHT'. "
        "A particularly high priority of 50 is designated for this continuous task, underscoring its operational importance. "
        "The crucial 'baseline_autonomy_rso' parameter, which specifies catalog IDs for tracking, should be populated with the string 'catID_A,catID_B,catID_C'. "
        "This objective is intended to run indefinitely to maintain foundational SSA, and therefore, no specific objective_end_time is to be specified, allowing it to run continuously. "
        "The 'rso_id_list', which will be dynamically populated at runtime, should be initialized to include 'rsoPlaceholder1' and 'rsoPlaceholder2' as a starting point."
    ): {
        "objective_uuid": "baseline-auto-uuid-987",
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 50,
        "baseline_autonomy_rso": "catID_A,catID_B,catID_C",
        "objective_end_time": None,
        "rso_id_list": ["rsoPlaceholder1", "rsoPlaceholder2"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 300: CatalogMaintenanceObjective
    (
        "Compose a CatalogMaintenanceObjective aimed at refining the catalogue for the MEO regime. This objective shall utilise sensors UKR08 and UKR09, operating in SIMULATED data mode under a Classification Marking of 'C'. Set the priority level to 500, with a generous patience of 45 minutes should intents encounter delays. The objective is scheduled to commence precisely at 2025-06-01 10:00:00+00:00 and conclude at 2025-06-01 18:00:00+00:00. Pray ensure that a binning setting of 4 is applied to the imager instrument parameters. We shall not require a visibility check prior to intent generation for this particular task."
    ): {
        "objective_name": "CatalogMaintenanceObjective",
        "classification_marking": "C",
        "data_mode": "SIMULATED",
        "orbital_regime": "MEO",
        "sensor_name_list": ["UKR08", "UKR09"],
        "priority": 500,
        "patience_minutes": 45,
        "objective_start_time": "datetime.datetime(2025, 6, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 1, 18, 0, tzinfo=TzInfo(UTC))",
        "binning": 4,
        "visibility_check": False,
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default but good to include for clarity
        "frame_type": "LIGHT",  # Default
        "end_time_offset_minutes": 20,  # Default
        "rso_id_list": [],  # Explicitly empty
        "objective_uuid": None,  # Let the system generate
    },
    # New Example 301: SearchObjective
    (
        "Initiate a SearchObjective targeting object 98765, employing the UKR15 sensor. This mission requires an 'S' classification marking and is to be executed in EXERCISE data mode. The search pattern shall be of the 'CROSS_TRACK' variety. The objective is set to begin at 2025-06-15 14:00:00+00:00 and terminate at 2025-06-15 16:00:00+00:00. Configure the initial offset to 90 seconds before the RSO's state and the final offset to 120 seconds thereafter. A frame overlap of 85 percent is deemed necessary for comprehensive coverage. Additionally, specify precisely 10 frames per intent, each with an integration time of 3.5 seconds. The priority for this search is exceptionally high, set to 1, and a visibility check is not required."
    ): {
        "objective_name": "SearchObjective",
        "classification_marking": "S",
        "target_id": "98765",
        "sensor_name": "UKR15",
        "search_type": "CROSS_TRACK",
        "data_mode": "EXERCISE",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 16, 0, tzinfo=TzInfo(UTC))",
        "initial_offset": 90,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "number_of_frames": 10,
        "integration_time": 3.5,
        "priority": 1,
        "visibility_check": False,
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "end_time_offset_minutes": 20,  # Default
        "binning": None,  # Optional, not specified
        "objective_uuid": None,  # Let the system generate
        "search_start_time": "datetime.datetime(2025, 6, 15, 14, 15, tzinfo=TzInfo(UTC))",  # Calculated automatically, but helpful in prompt
    },
    # New Example 302: GeodssRevisitObjective
    (
        "Create a GeodssRevisitObjective for targets 33445 and 77889, utilising sensors UKR21 and UKR22. This objective bears a 'TS' classification marking and operates in TEST data mode. The revisit patience should be set to 60 minutes, allowing ample time for execution. Instruct the system to ignore intent submissions from other objectives for these targets during the specified period. The objective is to commence at 2025-07-01 20:00:00+00:00. Configure the CCD array kind to 'Photometer' (1), the binning mode to 'No Binning' (0), and the scan mode to 'Continuous' (0). Set the gain setting to 'Low Gain' (1) and the SOI filter position to '10% Light' (2). Do not require a visibility check."
    ): {
        "objective_name": "GeodssRevisitObjective",
        "classification_marking": "TS",
        "target_id_list": ["33445", "77889"],
        "sensor_name_list": ["UKR21", "UKR22"],
        "data_mode": "TEST",
        "patience_minutes": 60,
        "ignore_other_objective_intent_submissions": True,
        "objective_start_time": "datetime.datetime(2025, 7, 1, 20, 0, tzinfo=TzInfo(UTC))",
        "array_kind": 1,
        "binning_mode": 0,
        "scan_mode": 0,
        "gain_setting": 1,
        "soi_filter_position": 2,
        "visibility_check": False,
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "priority": 10,  # Default
        "revisits_per_hour": None,  # Optional, not specified
        "number_of_frames": None,  # Optional, not specified
        "integration_time": None,  # Optional, not specified
        "objective_uuid": None,  # Let the system generate
        "objective_end_time": None,  # Optional, not specified
        "intent_start_time": None,  # Optional, not specified
        "intent_end_time": None,  # Optional, not specified
        "optimal_frames_per_hour": 400,  # Default
        "acquisition_type": 0,  # Default
        "auto_track_type": 0,  # Default (No Autotrack)
        "auto_track_roi_position": 0,  # Default
        "camera_mode": 0,  # Default
        "observation_interval": 0.0,  # Default
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "readout_rate_setting": 1,  # Default (2MHz)
        "rate_track_verify": 0,  # Default
        "overscan": 0,  # Default
        "command": 0,  # Default (Configure)
    },
    # New Example 303: PeriodicRevisitObjective
    (
        "Formulate a PeriodicRevisitObjective for targets 44556 and 88990, designating sensors UKR25 and UKR26 for the task. This objective necessitates a 'U//FOUO' classification marking and shall operate in REAL data mode. We require a revisit rate of 8 times per hour for each specified target. The objective is slated to commence on 2025-07-15 at 11:00:00+00:00. Set the patience for individual intents to 20 minutes. Furthermore, ensure that a visibility check is performed before generating each intent. The earliest time for intent submission should be 2025-07-15 11:30:00+00:00, and the latest time should be 2025-07-15 23:00:00+00:00. The priority for this objective is set to 5."
    ): {
        "objective_name": "PeriodicRevisitObjective",
        "classification_marking": "U//FOUO",
        "target_id_list": ["44556", "88990"],
        "sensor_name_list": ["UKR25", "UKR26"],
        "data_mode": "REAL",
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2025, 7, 15, 11, 0, tzinfo=TzInfo(UTC))",
        "patience_minutes": 20,
        "visibility_check": True,
        "intent_start_time": "datetime.datetime(2025, 7, 15, 11, 30, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 7, 15, 23, 0, tzinfo=TzInfo(UTC))",
        "priority": 5,
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "number_of_frames": None,  # Optional, not specified
        "integration_time": None,  # Optional, not specified
        "binning": None,  # Optional, not specified
        "objective_end_time": None,  # Optional, not specified
        "ignore_other_objective_intent_submissions": False,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,  # Let the system generate
    },
    # New Example 304: UctObservationObjective
    (
        "Devise a UctObservationObjective for observing unspecified UCT RSOs within the LEO orbital regime, leveraging sensors UKR30 and UKR31. Assign a 'C' classification marking and mandate operation in SIMULATED data mode. This objective should aim for a revisit rate of 10 UCTs per hour and will maintain a patience of 40 minutes per intent. The objective is scheduled to commence at 2025-08-01 09:00:00+00:00 and is set to conclude at 2025-08-01 17:00:00+00:00. Do not sort the UCTs by brightness for this particular task, and visibility checking is not required."
    ): {
        "objective_name": "UctObservationObjective",
        "classification_marking": "C",
        "orbital_regime": "LEO",
        "sensor_name_list": ["UKR30", "UKR31"],
        "data_mode": "SIMULATED",
        "revisits_per_hour": 10.0,
        "patience_minutes": 40,
        "objective_start_time": "datetime.datetime(2025, 8, 1, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 8, 1, 17, 0, tzinfo=TzInfo(UTC))",
        "sort_by_brightest_uct": False,
        "visibility_check": False,
        "uct_rso_id_list": None,  # Observe all UCTs
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "number_of_frames": None,  # Optional, not specified
        "integration_time": None,  # Optional, not specified
        "binning": None,  # Optional, not specified
        "end_time_offset_minutes": 60,  # Default
        "objective_uuid": None,  # Let the system generate
        "priority": 10,  # Default
    },
    # New Example 305: SingleIntentObjective
    (
        "Construct a SingleIntentObjective for RSO 55667, utilising solely the UKR35 sensor. This objective carries a 'TS' classification marking and will operate in TEST data mode. The intent should consist of 8 frames, each with an integration time of 4.0 seconds. Set the binning parameter to 1. The earliest permissible start time for this specific intent is 2025-08-15 21:00:00+00:00, and the latest permissible end time is 2025-08-15 22:00:00+00:00. The objective itself is to commence at 2025-08-15 20:30:00+00:00. Assign a priority of 3 to this objective."
    ): {
        "objective_name": "SingleIntentObjective",
        "classification_marking": "TS",
        "rso_id": "55667",
        "sensor_name_list": ["UKR35"],
        "data_mode": "TEST",
        "number_of_frames": 8,
        "integration_time": 4.0,
        "binning": 1,
        "intent_start_time": "datetime.datetime(2025, 8, 15, 21, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 8, 15, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_start_time": "datetime.datetime(2025, 8, 15, 20, 30, tzinfo=TzInfo(UTC))",
        "priority": 3,
        "target_id": None,  # Optional, not specified
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "objective_uuid": None,  # Let the system generate
        "objective_end_time": None,  # Optional, not specified
    },
    # New Example 306: DataEnrichmentObjective
    (
        "Request a DataEnrichmentObjective with an 'S' classification marking, to be executed in SIMULATED data mode. This objective is designed to enrich data for up to 10 RSOs selected by the system, aiming for a revisit rate of 15 revisits per hour. The observation should employ the 'RATE_TRACK' collection request type. The objective is to initiate at 2025-09-01 13:00:00+00:00 and conclude precisely at 2025-09-01 20:00:00+00:00. Visibility checking is not required for this data enrichment activity. Assign this objective a priority of 30."
    ): {
        "objective_name": "DataEnrichmentObjective",
        "classification_marking": "S",
        "data_mode": "SIMULATED",
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "collect_request_type": "RATE_TRACK",
        "objective_start_time": "datetime.datetime(2025, 9, 1, 13, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 1, 20, 0, tzinfo=TzInfo(UTC))",
        "visibility_check": False,
        "priority": 30,
        "objective_uuid": None,  # Let the system generate
        "target_id_list": None,  # Let the system select targets
        "sensor_name_list": None,  # Let the system select sensors
        "frame_type": "LIGHT",  # Default
        "binning": None,  # Optional, not specified
        "intent_start_time": None,  # Optional, not specified
        "intent_end_time": None,  # Optional, not specified
    },
    # New Example 307: SensorCheckoutObjective
    (
        "Establish a SensorCheckoutObjective for the UKR40 sensor, focusing specifically on the XGEO orbital regime. This checkout is to be performed with a 'U//FOUO' classification marking and in TEST data mode. Set the patience for each intent to 50 minutes and aim for 0.5 revisits per hour to allow for thorough checks. Visibility checking should be disabled. Configure the binning setting to 2. The objective is scheduled to run from 2025-09-15 10:00:00+00:00 until 2025-09-15 11:30:00+00:00. Assign a moderate priority of 15."
    ): {
        "objective_name": "SensorCheckoutObjective",
        "classification_marking": "U//FOUO",
        "sensor_name": "UKR40",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "patience_minutes": 50,
        "revisits_per_hour": 0.5,
        "visibility_check": False,
        "binning": 2,
        "objective_start_time": "datetime.datetime(2025, 9, 15, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 15, 11, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "number_of_frames": None,  # Optional, not specified
        "integration_time": None,  # Optional, not specified
        "objective_uuid": None,  # Let the system generate
        "intent_start_time": None,  # Optional, not specified
        "intent_end_time": None,  # Optional, not specified
    },
    # New Example 308: BaselineAutonomyObjective
    (
        "Define a BaselineAutonomyObjective identified by the UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. This objective shall operate with a 'C' classification marking and in EXERCISE data mode. It is primarily concerned with tracking catalog IDs '24680' and '13579'. The objective is intended to run continuously without a specified end time. Assign a priority level of 900. Ensure that the RSO ID list, which will be populated at runtime, is represented in the objective definition."
    ): {
        "objective_name": "BaselineAutonomyObjective",
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "C",
        "data_mode": "EXERCISE",
        "baseline_autonomy_rso": "24680,13579",
        "objective_end_time": None,
        "priority": 900,
        "frame_type": "LIGHT",  # Default
        "rso_id_list": None,  # Will be populated at runtime
    },
    # New Example 309: CatalogMaintenanceObjective
    (
        "hey can u make a CatalogMaintenanceObjective for me plz? i need it to use sensors RME55 and LMNT48, "
        "with S marking for secure stuff, and set it to REAL mode. make the priority 8 and set patience to like 15 mins. "
        "i want it to start tomorrow at 2025-05-17 03:25:00+00:00 and run for like 3 hours. use RATE_TRACK_SIDEREAL tracking "
        "in GEO regime and set visibility check to true. oh and add the RSO ID list with '33456,78123,90432'."
    ): {
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 5, 17, 6, 25, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 5, 17, 3, 25, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 15,
        "priority": 8,
        "rso_id_list": ["33456", "78123", "90432"],
        "sensor_name_list": ["RME55", "LMNT48"],
        "visibility_check": True,
    },
    # New Example 310: SearchObjective
    (
        "yo i need a search objective asap! target is 54321, and use sensor UKR35. make it U//FOUO classified and TEST mode. "
        "set priority to 3 and use SIDEREAL tracking. start it on 2025-06-10 at 14:15:00+00:00 and end at 2025-06-10 17:30:00+00:00. "
        "initial offset should be 45 seconds and final offset at 75 seconds with 65% frame overlap. do a CROSS_TRACK search type "
        "and make the search start 20 minutes after objective start."
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 20,
        "final_offset": 75,
        "frame_overlap_percentage": 0.65,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "objective_end_time": "datetime.datetime(2025, 6, 10, 17, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 14, 15, tzinfo=TzInfo(UTC))",
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 6, 10, 14, 35, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "UKR35",
        "target_id": "54321",
        "visibility_check": False,
    },
    # New Example 311: GeodssRevisitObjective
    (
        "make me a geodss revisit objective for targets 22334 and 55667 using sensors RME28 and LMNT30. set it to TS classification "
        "with real mode and priority 7. make it start on july 4th at 2025-07-04 22:00:00+00:00. set readout_rate to 0 (1MHz), gain to 1 (Low Gain), "
        "soi filter to 2 (10% Light), auto track type to 2 (Manual), camera mode to 1 (Zoomed EBS), array kind to 0 (Main), "
        "binning mode to 0 (No Binning), and scan mode to 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 0,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 4, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "patience_minutes": 30,
        "priority": 7,
        "readout_rate_setting": 0,
        "scan_mode": 0,
        "sensor_name_list": ["RME28", "LMNT30"],
        "soi_filter_position": 2,
        "target_id_list": ["22334", "55667"],
        "visibility_check": False,
    },
    # New Example 312: PeriodicRevisitObjective
    (
        "can u setup a periodic revisit objective? need to track targets 98765 and 43210 with sensors RME10 and LMNT11. "
        "give it c classification level and simulated data mode. set priority to 5, patience to 40 min, and make it do 8 revisits per hour. "
        "will need 3 frames each time with 1.5 sec integration. start on 2025-08-15 at 12:00:00+00:00 and let it run without end time. thx!"
    ): {
        "classification_marking": "C",
        "target_id_list": ["98765", "43210"],
        "sensor_name_list": ["RME10", "LMNT11"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 40,
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "objective_start_time": "datetime.datetime(2025, 8, 15, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 313: UctObservationObjective
    (
        "need to make a uct observation objective for these uct objects: 54321, 67890, 12345. we'll use sensors RME42 and LMNT44 "
        "with s classification in test mode. make it geo regime and priority 8 with 4 revisits per hour. start it on 2025-09-30 at 08:45:00+00:00, "
        "end after 12 hours, end time offset of 30 mins. need 4 frames each with 3 seconds integration. and pls enable sort by brightest uct."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "integration_time": 3,
        "number_of_frames": 4,
        "objective_end_time": "datetime.datetime(2025, 9, 30, 20, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "UctObservationObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 30, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 30,
        "priority": 8,
        "revisits_per_hour": 4.0,
        "sensor_name_list": ["RME42", "LMNT44"],
        "sort_by_brightest_uct": True,
        "uct_rso_id_list": ["54321", "67890", "12345"],
        "visibility_check": True,
    },
    # New Example 314: SingleIntentObjective
    (
        "i just need a quick single intent objective for target 44556 with rso id 88112. use sensors RME60 and LMNT62. "
        "make it u//fouo classification in exercise mode with rate_track tracking. set priority to 15, frames to 6, "
        "integration time to 2.5 seconds, and binning to 4. should start tomorrow at 2025-05-17 16:30:00+00:00."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "44556",
        "rso_id": "88112",
        "sensor_name_list": ["RME60", "LMNT62"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 6,
        "integration_time": 2.5,
        "priority": 15,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 5, 17, 16, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 315: DataEnrichmentObjective
    (
        "setup a data enrichment objective for me with targets 11223, 33445, 55667, 77889. use sensors RME75 and LMNT77 "
        "with ts classification in real mode. set max rso to observe to 10 and do 15 revisits per hour. start on 2025-06-05 at 14:00:00+00:00 "
        "and run for 48 hours. set priority to 25 and make sure visibility check is on."
    ): {
        "classification_marking": "TS",
        "data_mode": "REAL",
        "target_id_list": ["11223", "33445", "55667", "77889"],
        "sensor_name_list": ["RME75", "LMNT77"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 6, 5, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 6, 7, 14, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 316: SensorCheckoutObjective
    (
        "create a sensor checkout objective for sensor RME88. use u//fouo classification, real mode, meo regime, "
        "rate_track tracking, priority 12, and 2 revisits per hour. start it on 2025-07-20 at 10:00:00+00:00, run for 6 hours, "
        "with visibility check on. need 7 frames with 1.8 seconds integration time, and binning set to 3."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "RME88",
        "orbital_regime": "MEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 30,
        "revisits_per_hour": 2.0,
        "number_of_frames": 7,
        "integration_time": 1.8,
        "binning": 3,
        "objective_start_time": "datetime.datetime(2025, 7, 20, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 7, 20, 16, 0, tzinfo=TzInfo(UTC))",
        "priority": 12,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 317: BaselineAutonomyObjective
    (
        "need a baseline autonomy objective with uuid '987f6543-ab12-09d8-e765-432187654321'. use c classification, test mode, "
        "dark frame type, priority 900. set baseline autonomy rso to catalog ids '22446,88664,55113' and rso id list to "
        "'11335,77553,99775'. don't set an end time cuz we want it to run continuously."
    ): {
        "objective_uuid": "987f6543-ab12-09d8-e765-432187654321",
        "classification_marking": "C",
        "data_mode": "TEST",
        "frame_type": "DARK",
        "priority": 900,
        "baseline_autonomy_rso": "22446,88664,55113",
        "objective_end_time": None,
        "rso_id_list": ["11335", "77553", "99775"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 318: CatalogMaintenanceObjective
    (
        "yo, can you whip up a catalogmaintenanceobjective? make it for sensors 'xps01' and 'ytr05'. "
        "the classification marking should be 'C', and let's go with 'SIMULATED' for the data mode. "
        "priority needs to be, like, 1000, super low. patience is 45 minutes, and that end time offset thingy should be 30 minutes. "
        "oh, and make sure visibility check is true, k? "
        "this whole thing needs to start on october 10th, 2025, at 10 am utc, and wrap up same day at 3:30 pm utc. "
        "for tracking, use 'RATE_TRACK' and the orbital regime is 'MEO'. "
        "the rso id list should have '98765' and '54321'."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 10, 10, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 10, 10, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 1000,
        "rso_id_list": ["98765", "54321"],
        "sensor_name_list": ["xps01", "ytr05"],
        "visibility_check": True,
    },
    # New Example 319: SearchObjective
    (
        "alright, let's create a searchobjective. the target id is 'target_alpha_77', and we'll use sensor 'SENSRX99'. "
        "classification marking is 'TS', and data mode is 'EXERCISE'. "
        "priority is gonna be super high, set it to 1. the collect request type needs to be 'SIDEREAL'. "
        "it's gotta start on november 15th, 2025, at 08:00:00 utc. the end time is the same day, but at 09:00:00 utc. "
        "for the offsets, initial is 45 seconds, final is 75 seconds. frame overlap percentage should be 0.6. "
        "end time offset in minutes is 33. use 'CROSS_TRACK' for search type. "
        "and the search start time? make it 10 minutes after the objective start time. "
        "let's also specify 10 frames and an integration time of 0.5 seconds. visibility check can be false."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 33,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 0.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 11, 15, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 11, 15, 8, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "search_start_time": "datetime.datetime(2025, 11, 15, 8, 10, tzinfo=TzInfo(UTC))",  # 08:00 + 10 mins
        "search_type": "CROSS_TRACK",
        "sensor_name": "SENSRX99",
        "target_id": "target_alpha_77",
        "visibility_check": False,
    },
    # New Example 320: GeodssRevisitObjective
    (
        "okay, time for a geodssrevisitobjective. the targets are 'geo_target_001' and 'geo_target_002'. "
        "use sensors 'GEOEYE01' and 'GEOEYE02' for this one. marking is 'U//FOUO', and the mode is 'REAL'. "
        "set the priority to 7. tracking should be 'RATE_TRACK_SIDEREAL'. "
        "start date and time: december 1st, 2025, midnight utc. "
        "we need some really specific settings here: readout_rate is 0, which means 1mhz. gain_setting is 1 for low gain. "
        "soi_filter should be 2, that's 10% light. auto_track_type is 2 (manual). camera_mode is 1 (zoomed ebs). "
        "array_kind is 1 (photometer). binning_mode is 0, so no binning. scan_mode is 0 for continuous. "
        "also, let's aim for 5 revisits per hour and the number of frames should be 3. patience is 20 minutes."
    ): {
        "acquisition_type": 0,  # Default
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default
        "num_skip_frames": 0,  # Default
        "number_of_frames": 3,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 12, 1, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 20,
        "priority": 7,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": 5.0,
        "scan_mode": 0,
        "sensor_name_list": ["GEOEYE01", "GEOEYE02"],
        "soi_filter_position": 2,
        "target_id_list": ["geo_target_001", "geo_target_002"],
        "visibility_check": False,  # Default
    },
    # New Example 321: PeriodicRevisitObjective
    (
        "next up, a periodicrevisitobjective. the list of target ids includes 'periodic_obj_1' and 'periodic_obj_2'. "
        "sensors for this will be 'PRD01' and 'PRD02'. marking is 'S', and the data mode is 'TEST'. "
        "priority is 3. patience should be 25 minutes. and yeah, set ignore other objective submissions to true. "
        "the objective starts on january 15, 2026, at 12:00:00 utc. "
        "optimal frames per hour is 300, number of frames is 3, and integration time is 1.5 seconds. "
        "we also want 4.5 revisits per hour. make visibility check true for this one."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "data_mode": "TEST",
        "frame_type": "LIGHT",  # Default
        "ignore_other_objective_intent_submissions": True,
        "integration_time": 1.5,
        "intent_end_time": None,
        "intent_start_time": None,
        "number_of_frames": 3,
        "objective_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
        "objective_start_time": "datetime.datetime(2026, 1, 15, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "optimal_frames_per_hour": 300,
        "patience_minutes": 25,
        "priority": 3,
        "revisits_per_hour": 4.5,
        "sensor_name_list": ["PRD01", "PRD02"],
        "target_id_list": ["periodic_obj_1", "periodic_obj_2"],
        "visibility_check": True,
    },
    # New Example 322: UctObservationObjective
    (
        "let's get a uctobservationobjective going. the uct rso ids are 'uct_sat_delta' and 'uct_sat_gamma'. "
        "sensors: 'UCTSEN1', 'UCTSEN2'. marking is 'U//FOUO', mode is 'REAL'. "
        "this is for the 'LEO' orbital regime. priority is 8, and we need 5 revisits per hour. "
        "start this on february 20, 2026, at 3 in the morning utc. "
        "make sure sort by brightest uct is false. end time offset is 40 minutes. "
        "visibility check should be true. number of frames is 4, integration time 2.5 seconds. set binning to 1."
    ): {
        "binning": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "data_mode": "REAL",
        "end_time_offset_minutes": 40,
        "frame_type": "LIGHT",  # Default
        "integration_time": 2.5,
        "number_of_frames": 4,
        "objective_end_time": None,
        "objective_name": "UctObservationObjective",
        "objective_start_time": "datetime.datetime(2026, 2, 20, 3, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "LEO",
        "patience_minutes": 30,  # Default
        "priority": 8,
        "revisits_per_hour": 5.0,
        "sensor_name_list": ["UCTSEN1", "UCTSEN2"],
        "sort_by_brightest_uct": False,
        "uct_rso_id_list": ["uct_sat_delta", "uct_sat_gamma"],
        "visibility_check": True,
    },
    # New Example 323: SingleIntentObjective
    (
        "now, how about a singleintentobjective. target id is 'single_t_42', and the rso id is 'single_r_007'. "
        "the sensors to use are 'SNGLSENS1' and 'SNGLSENS2'. "
        "marking is 'U'. data mode is 'SIMULATED'. the collect request type is 'RATE_TRACK'. priority is 1. "
        "objective start time should be march 10th, 2026, at 06:00:00 utc. "
        "number of frames: 2. integration time: 3 seconds. binning is 4. "
        "also, let's give it an objective end time: same day, march 10th, 2026, at 06:30:00 utc."
    ): {
        "binning": 4,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "frame_type": "LIGHT",  # Default
        "integration_time": 3.0,
        "intent_end_time": None,
        "intent_start_time": None,
        "number_of_frames": 2,
        "objective_end_time": "datetime.datetime(2026, 3, 10, 6, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
        "objective_start_time": "datetime.datetime(2026, 3, 10, 6, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "rso_id": "single_r_007",
        "sensor_name_list": ["SNGLSENS1", "SNGLSENS2"],
        "target_id": "single_t_42",
    },
    # New Example 324: DataEnrichmentObjective
    (
        "it's dataenrichmentobjective time! so, the target ids are 'data_e_101' and 'data_e_102'. "
        "for sensors, use 'ENRICH01' and 'ENRICH02'. the classification marking is 'C'. "
        "data mode will be 'EXERCISE'. collect request type needs to be 'SIDEREAL'. "
        "max rso to observe is 10. we want 15 revisits per hour. "
        "this starts on april 5th, 2026, at 9 am utc. visibility check should be false. "
        "priority is 15. and let's end this objective on the same day at 6 pm utc."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",  # Default
        "intent_end_time": None,
        "intent_start_time": None,
        "max_rso_to_observe": 10,
        "objective_end_time": "datetime.datetime(2026, 4, 5, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "DataEnrichmentObjective",
        "objective_start_time": "datetime.datetime(2026, 4, 5, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 15,
        "revisits_per_hour": 15.0,
        "sensor_name_list": ["ENRICH01", "ENRICH02"],
        "target_id_list": ["data_e_101", "data_e_102"],
        "visibility_check": False,
    },
    # New Example 325: SensorCheckoutObjective
    (
        "sensorcheckoutobjective, comin' right up! the marking is 'TS'. sensor name is 'CHKOUTSENS7'. "
        "data mode is 'REAL'. orbital regime is 'XGEO'. use 'RATE_TRACK_SIDEREAL' for tracking. "
        "priority 5. revisits per hour should be 0.5. "
        "start time is may 1st, 2026, exactly at midnight utc. "
        "visibility check is false. patience is 50 minutes. "
        "number of frames is 6, and integration time is 1.0 second. binning is 2."
    ): {
        "binning": 2,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "frame_type": "LIGHT",  # Default
        "integration_time": 1.0,
        "intent_end_time": None,
        "intent_start_time": None,
        "number_of_frames": 6,
        "objective_end_time": None,
        "objective_name": "SensorCheckoutObjective",
        "objective_start_time": "datetime.datetime(2026, 5, 1, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "XGEO",
        "patience_minutes": 50,
        "priority": 5,
        "revisits_per_hour": 0.5,
        "sensor_name": "CHKOUTSENS7",
        "visibility_check": False,
    },
    # New Example 326: BaselineAutonomyObjective
    (
        "and last but not least, a baselineautonomyobjective. the uuid is 'abcdef01-2345-6789-abcd-ef0123456789'. "
        "marking is 'U//FOUO'. data mode is 'SIMULATED'. frame type is 'DARK'. "
        "priority is gonna be 500, pretty high. "
        "the baseline autonomy rso string should be 'cat_id_001,cat_id_002,cat_id_003'. "
        "let's set an objective end time for december 31st, 2026, at 23:59:59 utc. "
        "the rso id list includes 'rso_x' and 'rso_y'."
    ): {
        "baseline_autonomy_rso": "cat_id_001,cat_id_002,cat_id_003",
        "classification_marking": "U//FOUO",
        "data_mode": "SIMULATED",
        "frame_type": "DARK",
        "objective_end_time": "datetime.datetime(2026, 12, 31, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "objective_name": "BaselineAutonomyObjective",
        "objective_uuid": "abcdef01-2345-6789-abcd-ef0123456789",
        "priority": 500,
        "rso_id_list": ["rso_x", "rso_y"],
    },
    # New Example 327: CatalogMaintenanceObjective
    (
        "hey, let's set up a catalog maintenance thingy, ok? wanna use sensors bigeye-1 and sharpview-a, and let's slap a c marking on it. "
        "we'll run this in real mode, with a kinda low priority, like 500. we can wait 45 minutes if something goes wonky. "
        "schedule it to start pretty soon, say 2025-06-01 10:00:00+00:00, and end it at 2025-06-01 11:00:00+00:00. "
        "let's focus on the geo regime for this one, using the default tracking type. "
        "oh, and the specific rso ids we care about are '98765, 101112, 131415'. don't bother with the visibility check, just go for it."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "data_mode": "REAL",
        "end_time_offset_minutes": 20,  # default
        "frame_type": "LIGHT",  # default
        "objective_end_time": "datetime.datetime(2025, 6, 1, 11, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 45,
        "priority": 500,
        "rso_id_list": ["98765", "101112", "131415"],
        "sensor_name_list": ["bigeye-1", "sharpview-a"],
        "visibility_check": False,
    },
    # New Example 328: SearchObjective
    (
        "hey, could you whip up a search objective for target abcd-efgh? wanna use the spyglass-x sensor. "
        "mark it with s, and run it in simulated mode. use the default rate track type. "
        "start the whole shebang at 2025-06-15 14:00:00+00:00 and wrap it up by 2025-06-15 15:30:00+00:00. "
        "for the search, start 90 seconds before the target's state and end 120 seconds after. "
        "make sure the frames overlap by like, 85 percent. the end time offset for scheduling should be 30 minutes. "
        "use an cross_track search pattern, and the search should actually start 20 minutes after the objective begins. "
        "don't need to check visibility beforehand."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 30,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",  # default
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 6, 15, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,  # default
        "search_start_time": "datetime.datetime(2025, 6, 15, 14, 20, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "spyglass-x",
        "target_id": "abcd-efgh",
        "visibility_check": False,
    },
    # New Example 329: GeodssRevisitObjective
    (
        "set up a geodss revisit objective for targets target-alpha and target-beta. "
        "use sensors site22-geodss and site23-geodss. this is for test mode. "
        "put a ts marking on it. keep the default rate track tracking. "
        "start the objective at 2025-07-07 20:00:00+00:00. we want a patience of 40 minutes. "
        "set the readout rate to 1mhz (that's setting 0) and the gain to low gain (setting 1). "
        "use the open soi filter position (setting 0). set the auto track type to manual (setting 2). "
        "camera mode should be normal (setting 0). use the main array (setting 0). "
        "no binning needed (setting 0). scan mode should be single frame (setting 1). "
        "set number of observations to 2. no frames or integration time specified."
    ): {
        "acquisition_type": 0,  # default
        "array_kind": 0,
        "auto_track_roi_position": 0,  # default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 0,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "command": 0,  # default
        "data_mode": "TEST",
        "frame_type": "LIGHT",  # default
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 2,
        "num_skip_frames": 0,  # default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 7, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # default
        "optimal_frames_per_hour": 400,  # default
        "overscan": 0,  # default
        "patience_minutes": 40,
        "priority": 10,  # default
        "rate_track_verify": 0,  # default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 1,
        "sensor_name_list": ["site22-geodss", "site23-geodss"],
        "soi_filter_position": 0,
        "target_id_list": ["target-alpha", "target-beta"],
        "visibility_check": False,  # default
    },
    # New Example 330: PeriodicRevisitObjective
    (
        "create a periodic revisit objective for targets pqr-123 and xyz-456 using the newton-a sensor. "
        "this needs a u//fouo marking and should be in real mode. "
        "set the priority to 5. patience can be the default 30 minutes. "
        "start the objective at 2025-08-20 03:00:00+00:00. "
        "let's get 8 revisits per hour for each target. "
        "capture 3 frames each time, with an integration time of 1.5 seconds. "
        "no binning. don't ignore other objective submissions. optimal frames per hour is the standard 400."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["pqr-123", "xyz-456"],
        "sensor_name_list": ["newton-a"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "frame_type": "LIGHT",  # default
        "visibility_check": False,  # default
        "patience_minutes": 30,  # default
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 8, 20, 3, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "ignore_other_objective_intent_submissions": False,  # default
        "optimal_frames_per_hour": 400,  # default
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 331: UctObservationObjective
    (
        "hey, can you make a uct observation objective for uct rsos 77889, 22334, and 55667? "
        "use the skymaster-z sensor. it's got a u marking. "
        "we'll run this in exercise mode. focus on the meo regime. "
        "we want about 4 revisits per hour. start this objective at 2025-09-05 21:00:00+00:00. "
        "make sure to enable sorting by the brightest uct first. "
        "the end time offset should be 90 minutes. definitely do the visibility check. "
        "let's get 10 frames per observation with an integration time of 3 seconds. default patience is fine."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["77889", "22334", "55667"],
        "sensor_name_list": ["skymaster-z"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "frame_type": "LIGHT",  # default
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 30,  # default
        "revisits_per_hour": 4.0,
        "number_of_frames": 10,
        "integration_time": 3.0,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 10,  # default
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 332: SingleIntentObjective
    (
        "i need a single intent objective for target uuid 998877 and rso id 112233. "
        "gonna use the globescan-g sensor for this. "
        "use a s marking and real data mode. standard rate track tracking is good. "
        "priority 3 is important for this one. "
        "start the objective at 2025-10-10 01:00:00+00:00. "
        "capture 7 frames with an integration time of 0.5 seconds. "
        "set binning to 4. intent times can be decided automatically."
    ): {
        "classification_marking": "S",
        "target_id": "998877",
        "rso_id": "112233",
        "sensor_name_list": ["globescan-g"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "frame_type": "LIGHT",  # default
        "number_of_frames": 7,
        "integration_time": 0.5,
        "priority": 3,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 1, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 333: DataEnrichmentObjective
    (
        "let's do a data enrichment objective for targets data-target-1, data-target-2, and data-target-3. "
        "we'll use sensors rme44 and lmnt45. "
        "put a c marking on this one. run in real mode. "
        "use the rate_track collect type. "
        "we want to observe a maximum of 10 rsos. aim for 15 revisits per hour. "
        "start the objective on 2025-11-18 18:00:00+00:00. "
        "definitely do the visibility check for this. priority should be the default."
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["data-target-1", "data-target-2", "data-target-3"],
        "sensor_name_list": ["rme44", "lmnt45"],
        "collect_request_type": "RATE_TRACK",  # specified
        "frame_type": "LIGHT",  # default
        "binning": None,  # default
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 11, 18, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,  # default
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 334: SensorCheckoutObjective
    (
        "okay, time for a sensor checkout objective. target sensor vista-b. "
        "classification should be u//fouo. run in test mode. "
        "focus on the xgeo regime. use the default rate track tracking type. "
        "set the priority to 5. patience is the standard 30 mins. "
        "we need 2 revisits per hour for the checkout. "
        "start the objective at 2026-01-01 05:00:00+00:00. "
        "capture 8 frames with 2.5 seconds of integration time each. "
        "make sure visibility check is on."
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "vista-b",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # default
        "frame_type": "LIGHT",  # default
        "visibility_check": True,
        "patience_minutes": 30,  # default
        "revisits_per_hour": 2.0,
        "number_of_frames": 8,
        "integration_time": 2.5,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 1, 5, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 335: BaselineAutonomyObjective
    (
        "create a baseline autonomy objective. the uuid for this guy is 'fedcba98-7654-3210-fedc-ba9876543210'. "
        "it needs a u marking and should operate in real mode. "
        "frame type is light. priority should be really high, like 500. "
        "the main catalog ids for baseline tracking are '24680,13579'. "
        "also include rso ids 'alpha-1, beta-2, gamma-3' in the list. "
        "this one should run continuously, so no end time."
    ): {
        "objective_uuid": "fedcba98-7654-3210-fedc-ba9876543210",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 500,
        "baseline_autonomy_rso": "24680,13579",
        "objective_end_time": None,
        "rso_id_list": ["alpha-1", "beta-2", "gamma-3"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 336: CatalogMaintenanceObjective
    (
        "WHOA! Set up a SUPER COOL CatalogMaintenanceObjective for sensors BLINK42 and FLASH99 with TOP SECRET classification! "
        "We're in REAL mode with a MEGA priority of 5 and patience of just 15 minutes! "
        "Set end time offset to a ZIPPY 30 minutes and check visibility - YES PLEASE! "
        "Start the mission on 2025-06-15 at 14:30:00+00:00 and wrap it up by 2025-06-15 at 20:45:00+00:00! "
        "Use RATE_TRACK_SIDEREAL tracking in the AWESOME GEO regime with RSO IDs '55123,78456,99002'!"
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 20, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 15,
        "priority": 5,
        "rso_id_list": ["55123", "78456", "99002"],
        "sensor_name_list": ["BLINK42", "FLASH99"],
        "visibility_check": True,
    },
    # New Example 337: SearchObjective
    (
        "POW! Time for a RADICAL SearchObjective targeting ID 54321 with the AWESOME sensor ZAP55! "
        "Use the CONFIDENTIAL marking with SIMULATED mode (for practice, dudes!). Set a ZOOMING priority of 3! "
        "We're going with RATE_TRACK tracking because we're EXTREME! Start searching on 2025-08-10 at 06:15:00+00:00 "
        "and finish by 2025-08-10 at 09:45:00+00:00! Initial offset is a QUICK 45 seconds, final offset is 120 seconds, "
        "with a MASSIVE 80% frame overlap! End time offset is 35 minutes and we're using CROSS_TRACK search type with "
        "search starting 25 minutes after objective start. VISIBILITY CHECK ON!"
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 35,
        "final_offset": 120,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 8, 10, 9, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 10, 6, 15, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 8, 10, 6, 40, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "ZAP55",
        "target_id": "54321",
        "visibility_check": True,
    },
    # New Example 338: GeodssRevisitObjective
    (
        "KABOOM! Create a SPECTACULAR GeodssRevisitObjective for targets 33333,44444,55555 using our SUPERCOOL sensors BOOM01 and KAPOW02! "
        "Mark it U//FOUO with TEST mode and SIDEREAL tracking! Set a WACKY priority of 8! Start the fun on 2025-09-05 at 22:00:00+00:00! "
        "Configure it with readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), auto_track_type 2 (Manual), "
        "camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), and scan_mode 0 (Continuous)! "
        "Make patience a GENEROUS 45 minutes and number of frames a WHOPPING 10!"
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "command": 0,
        "data_mode": "TEST",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,
        "num_skip_frames": 0,
        "number_of_frames": 10,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 45,
        "priority": 8,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["BOOM01", "KAPOW02"],
        "soi_filter_position": 2,
        "target_id_list": ["33333", "44444", "55555"],
        "visibility_check": False,
    },
    # New Example 339: PeriodicRevisitObjective
    (
        "ZOWIE! Set up a DYNAMITE PeriodicRevisitObjective for targets 98765,43210,11223 using our FANTASTIC sensors ZOOM07,BLAST09! "
        "Make it TOP SECRET level with EXERCISE mode, and a BLAZING priority of 4! We need patience of 20 minutes and 8 revisits every hour! "
        "Start observing on 2025-07-10 at 16:45:00+00:00 and keep watching until 2025-07-11 at 16:45:00+00:00! "
        "Set visibility check ON, capture 8 frames each time with integration time of 3.5 seconds! "
        "Don't ignore other objective submissions, and set optimal frames per hour to a MIND-BLOWING 600!"
    ): {
        "classification_marking": "TS",
        "target_id_list": ["98765", "43210", "11223"],
        "sensor_name_list": ["ZOOM07", "BLAST09"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 8,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 7, 10, 16, 45, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 7, 11, 16, 45, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 600,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 340: UctObservationObjective
    (
        "BAZINGA! Time for an AMAZING UctObservationObjective to track those mysterious UCT RSOs 12121,34343,56565,78787! "
        "Deploy our INCREDIBLE sensors WHIZ22 and FLASH33! Mark it CONFIDENTIAL and run in REAL mode with SIDEREAL tracking! "
        "We're watching the MEO regime with a cool priority of 7! Keep patient for 25 minutes and aim for 5 revisits per hour! "
        "Start observing on 2025-10-15 at 03:30:00+00:00 with end time offset of 90 minutes! "
        "Take 6 frames each visit with integration time of 4 seconds! Visibility check ON and sort by brightest UCT? YOU BETCHA!"
    ): {
        "classification_marking": "C",
        "uct_rso_id_list": ["12121", "34343", "56565", "78787"],
        "sensor_name_list": ["WHIZ22", "FLASH33"],
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 25,
        "revisits_per_hour": 5.0,
        "number_of_frames": 6,
        "integration_time": 4,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 15, 3, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 341: SingleIntentObjective
    (
        "WHAMMO! Let's create a ONE-OF-A-KIND SingleIntentObjective with target ID 76543 and RSO ID 88888! "
        "Use our INCREDIBLE sensors POW44,BANG45,ZOOM46! Slap a SECRET classification on it with TEST mode! "
        "We'll use RATE_TRACK (not SIDEREAL, not RATE_TRACK_SIDEREAL - just plain RATE_TRACK!) with a SUPER priority of 6! "
        "Start at 2025-11-20 at 12:15:00+00:00 and end at 2025-11-20 at 14:45:00+00:00! "
        "Capture 12 frames with integration time of 1.5 seconds and binning of 3! LIGHT frame type, of course!"
    ): {
        "classification_marking": "S",
        "target_id": "76543",
        "rso_id": "88888",
        "sensor_name_list": ["POW44", "BANG45", "ZOOM46"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 12,
        "integration_time": 1.5,
        "priority": 6,
        "binning": 3,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 20, 12, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 20, 14, 45, tzinfo=TzInfo(UTC))",
        "objective_name": "SingleIntentObjective",
    },
    # New Example 342: DataEnrichmentObjective
    (
        "KAPOW! Time for a DATA-TASTIC DataEnrichmentObjective for targets 11111,22222,33333,44444,55555! "
        "Use our SUPER DUPER sensors FLASH77,ZOOM88,BANG99! U//FOUO classification with EXERCISE mode! "
        "We'll stick with the DEFAULT RATE_TRACK_SIDEREAL tracking and set max RSO to observe to a WHOPPING 12! "
        "Aim for 15 revisits per hour with priority 15! Start the data party on 2025-12-05 at 13:00:00+00:00! "
        "Keep visibility check TRUE because we're RESPONSIBLE space watchers!"
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["11111", "22222", "33333", "44444", "55555"],
        "sensor_name_list": ["FLASH77", "ZOOM88", "BANG99"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 12, 5, 13, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 343: SensorCheckoutObjective
    (
        "COWABUNGA! Let's run a FANTABULOUS SensorCheckoutObjective for our newly upgraded sensor TURBO55! "
        "We'll use an UNCLASSIFIED marking with REAL mode in the LEO regime! Set SIDEREAL tracking with priority 9! "
        "Keep visibility check ON, patience at 40 minutes, and aim for 2 revisits per hour! "
        "Start checking out on 2026-01-15 at 09:30:00+00:00 and continue until 2026-01-16 at 09:30:00+00:00! "
        "Take 7 frames each visit with integration time of 2.5 seconds and binning of 2!"
    ): {
        "classification_marking": "U",
        "sensor_name": "TURBO55",
        "orbital_regime": "LEO",
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.0,
        "number_of_frames": 7,
        "integration_time": 2.5,
        "binning": 2,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 9, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 16, 9, 30, tzinfo=TzInfo(UTC))",
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 9,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 344: BaselineAutonomyObjective
    (
        "HOLY SATELLITES! Create an ULTRA AWESOME BaselineAutonomyObjective with UUID '999e8888-f77b-66d5-c444-555588889999'! "
        "Let's use SECRET classification with REAL mode and LIGHT frame type! Set priority to a MASSIVE 800! "
        "Track these AMAZING catalog IDs: 28450, 35640, 42970, and 57890 along with RSO IDs: 12345, 67890, 13579, 24680! "
        "This objective is ALWAYS ON, so no end time needed - it's a NEVER-ENDING SPACE ADVENTURE!"
    ): {
        "objective_uuid": "999e8888-f77b-66d5-c444-555588889999",
        "classification_marking": "S",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 800,
        "baseline_autonomy_rso": "28450,35640,42970,57890",
        "objective_end_time": None,
        "rso_id_list": ["12345", "67890", "13579", "24680"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 345: CatalogMaintenanceObjective
    (
        "Kaboom! We need a CatalogMaintenanceObjective, like, yesterday! Let's get sensors 'COSMO-CAM-01' and 'STAR-GAZER-9000' on the job with 'S' classification markings. "
        "The data mode is 'SIMULATED' because we're practicing our awesome moves! This is super important, so set the priority to 10! "
        "We'll need a patience of 20 minutes, an end time offset of 40 minutes, and absolutely, positively set visibility check to true. "
        "Let's kick this off on 2025-07-15 10:00:00+00:00 and wrap it up by 2025-07-15 18:00:00+00:00. We're focusing on the 'GEO' orbital regime. "
        "Make sure to use 'RATE_TRACK' for the collect request type. The RSO ID list should be 'SATOPS-1122' and 'ASTRO-DOG-007'. Oh, and set binning to 2x2."
    ): {
        "binning": 2,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 40,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 7, 15, 18, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 15, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "GEO",
        "patience_minutes": 20,
        "priority": 10,
        "rso_id_list": ["SATOPS-1122", "ASTRO-DOG-007"],
        "sensor_name_list": ["COSMO-CAM-01", "STAR-GAZER-9000"],
        "visibility_check": True,
    },
    # New Example 346: SearchObjective
    (
        "Alright, space rangers! Time for a mega SearchObjective! Our prime target is 'UNKNOWN-X-FACTOR-99' using the one and only sensor 'SEARCHMASTER-PRO'. "
        "Slap a 'U//FOUO' marking on this one, data mode is 'EXERCISE', and the priority is a top-tier 2! "
        "We're going with 'SIDEREAL' tracking for this cosmic hunt. The mission starts precisely at 2025-08-20 08:30:00+00:00 and must end by 2025-08-20 14:45:00+00:00. "
        "Set the initial offset to a snappy 25 seconds, and the final offset to a generous 100 seconds. We need a frame overlap of 0.60 (that's 60 percent, folks!). "
        "The end time offset will be 35 minutes. Use the 'RASTER_SCAN' search type, and initiate the search 10 minutes after the objective's grand beginning. Visibility check is a must, so true it is! And let's get 10 frames with an integration time of 0.5 seconds."
    ): {
        "binning": None,
        "classification_marking": "U//FOUO",
        "collect_request_type": "SIDEREAL",
        "data_mode": "EXERCISE",
        "end_time_offset_minutes": 35,
        "final_offset": 100,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 25,
        "integration_time": 0.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 8, 20, 14, 45, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 20, 8, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 2,
        "search_start_time": "datetime.datetime(2025, 8, 20, 8, 40, 0, tzinfo=TzInfo(UTC))",  # Manually calculated as objective_start_time + 10 minutes
        "search_type": "RASTER_SCAN",  # Assuming RASTER_SCAN is a valid SearchType
        "sensor_name": "SEARCHMASTER-PRO",
        "target_id": "UNKNOWN-X-FACTOR-99",
        "visibility_check": True,
    },
    # New Example 347: GeodssRevisitObjective
    (
        "Cowabunga, dudes and dudettes! It's GeodssRevisitObjective time! We're locking onto targets 'GALAXY-GUARDIAN-1' and 'NEBULA-NAVIGATOR-5' with sensors 'GEO-TRACKER-ALPHA' and 'SKYNET-DELTA'. "
        "This one's got 'C' markings, the mode is 'REAL' deal, and priority is a cool 8. Tracking is 'RATE_TRACK_SIDEREAL'. "
        "Initiate operations at 2025-09-05 20:00:00+00:00. Let's get specific with the GEODSS settings: readout_rate is 0 (1MHz), gain_setting is 0 (High Gain), "
        "soi_filter is 0 (Open), auto_track_type is 0 (No Autotrack), camera_mode is 2 (Binned CCD), array_kind is 0 (Main), "
        "binning_mode is 1 (HW Binning), scan_mode is 0 (Continuous). We want 5 observations with a 0.2 second interval. Set patience to 40 minutes and make sure 'ignore_other_objective_intent_submissions' is false."
    ): {
        "acquisition_type": 0,  # Default
        "array_kind": 0,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 0,
        "binning_mode": 1,
        "camera_mode": 2,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 0,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 5,
        "num_skip_frames": 0,  # Default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 9, 5, 20, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.2,
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 40,
        "priority": 8,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEO-TRACKER-ALPHA", "SKYNET-DELTA"],
        "soi_filter_position": 0,
        "target_id_list": ["GALAXY-GUARDIAN-1", "NEBULA-NAVIGATOR-5"],
        "visibility_check": False,  # Default
    },
    # New Example 348: PeriodicRevisitObjective
    (
        "Get ready for a PeriodicRevisitObjective, team! We're keeping tabs on targets 'COMET-CRUISER-42' and 'ASTEROID-ACE-77' using sensors 'WIDEFIELD-IMAGER-Z' and 'PRECISION-SCOPE-Y'. "
        "This is a 'TS' classification, 'TEST' mode, with a priority of 4. Let's set patience to a swift 15 minutes. "
        "We want this to kick off on 2025-10-25 00:00:00+00:00 and run until 2025-10-26 00:00:00+00:00. "
        "We absolutely need 8 revisits per hour! Capture 3 frames per intent, with an integration time of 1.5 seconds. Set binning to 4. "
        "Oh, and 'ignore_other_objective_intent_submissions' should be true, we're going rogue! Optimal frames per hour can stay at default 400. Visibility check is false. Collect request type 'RATE_TRACK_SIDEREAL'."
    ): {
        "classification_marking": "TS",
        "target_id_list": ["COMET-CRUISER-42", "ASTEROID-ACE-77"],
        "sensor_name_list": ["WIDEFIELD-IMAGER-Z", "PRECISION-SCOPE-Y"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 15,
        "revisits_per_hour": 8.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": 4,
        "objective_start_time": "datetime.datetime(2025, 10, 25, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 26, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 4,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 349: UctObservationObjective
    (
        "Zoinks! A UctObservationObjective is what we need! For UCT RSOs 'UCT-STARLINK-FAKE-1' and 'UCT-ONEWEB-MOCK-2', deploy sensors 'UCT-HUNTER-101' and 'GHOST-WATCHER-00'. "
        "Markings are 'U', mode is 'SIMULATED', orbital regime 'LEO', priority a smashing 7! We desire 4.5 revisits per hour. "
        "Start this cosmic stakeout at 2025-11-30 13:30:00+00:00. Let's sort by the brightest UCT, absolutely! The end time offset should be 75 minutes, and visibility check must be true. "
        "We'll take 6 frames each time, with an integration time of 2.5 seconds. Patience is 25 minutes."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["UCT-STARLINK-FAKE-1", "UCT-ONEWEB-MOCK-2"],
        "sensor_name_list": ["UCT-HUNTER-101", "GHOST-WATCHER-00"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "LEO",
        "visibility_check": True,  # Default is True for this objective type
        "patience_minutes": 25,
        "revisits_per_hour": 4.5,
        "number_of_frames": 6,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 30, 13, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 350: SingleIntentObjective
    (
        "Booyah! Let's create a SingleIntentObjective! The target ID is 'TARGET-ALPHA-001', and the RSO ID is 'RSO-OMEGA-999'. Engage sensors 'PINPOINT-ARRAY-X' and 'FOCUS-MASTER-Z'. "
        "Classification marking 'S', data_mode 'REAL', tracking type 'RATE_TRACK'. Priority is a high-flying 3! "
        "This objective needs to start on 2025-12-24 06:00:00+00:00. We're going for 7 frames, an integration time of 3 seconds, and binning set to 1 (no actual binning, just 1x1). "
        "The intent itself should start no earlier than 2025-12-24 06:15:00+00:00 and end no later than 2025-12-24 07:00:00+00:00."
    ): {
        "classification_marking": "S",
        "target_id": "TARGET-ALPHA-001",
        "rso_id": "RSO-OMEGA-999",
        "sensor_name_list": ["PINPOINT-ARRAY-X", "FOCUS-MASTER-Z"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 7,
        "integration_time": 3.0,
        "priority": 3,
        "binning": 1,
        "intent_start_time": "datetime.datetime(2025, 12, 24, 6, 15, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": "datetime.datetime(2025, 12, 24, 7, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 24, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # This objective type doesn't typically have its own end time if intent times are specified
        "objective_name": "SingleIntentObjective",
    },
    # New Example 351: DataEnrichmentObjective
    (
        "Holy moly! It's DataEnrichmentObjective time, folks! We're enriching data for targets 'DATA-POINT-DELTA', 'INFO-BIT-GAMMA', and 'KNOWLEDGE-NUGGET-SIGMA'. "
        "Sensors 'ENRICHER-BOT-500' and 'DATAMINE-DRONE-7' are on it! This is 'U//FOUO' classified, 'EXERCISE' mode. "
        "Let's track with 'SIDEREAL' and observe a maximum of 10 RSOs. We're aiming for an amazing 15 revisits per hour! "
        "This mission kicks off on 2026-01-10 10:00:00+00:00 and ends on 2026-01-11 10:00:00+00:00. Priority is 25, 'cause data is precious but not THAT precious right now. "
        "Visibility check is definitely true for this. No binning needed for this one. And the intent should start at 2026-01-10 10:30:00+00:00."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": [
            "DATA-POINT-DELTA",
            "INFO-BIT-GAMMA",
            "KNOWLEDGE-NUGGET-SIGMA",
        ],
        "sensor_name_list": ["ENRICHER-BOT-500", "DATAMINE-DRONE-7"],
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2026, 1, 11, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "priority": 25,
        "intent_start_time": "datetime.datetime(2026, 1, 10, 10, 30, 0, tzinfo=TzInfo(UTC))",
        "intent_end_time": None,
        "visibility_check": True,  # Default is True for this objective type
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 352: SensorCheckoutObjective
    (
        "To infinity and beyond with this SensorCheckoutObjective! We're checking out sensor 'TESTUDO-CAM-007'. "
        "Markings are 'U', data_mode is 'TEST', and the orbital_regime is 'MEO'. This has a priority of 5. "
        "We're aiming for 2.5 revisits_per_hour, starting on 2026-02-14 09:00:00+00:00. Let's give it a patience of 20 minutes. "
        "Capture 4 frames with an integration_time of 1.0 second. Visibility_check is true. Collect request type is 'RATE_TRACK'. No binning this time."
    ): {
        "classification_marking": "U",
        "sensor_name": "TESTUDO-CAM-007",
        "orbital_regime": "MEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 20,
        "revisits_per_hour": 2.5,
        "number_of_frames": 4,
        "integration_time": 1.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 14, 9, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 353: BaselineAutonomyObjective
    (
        "Gadzooks! It's a BaselineAutonomyObjective! Generate it with the flashy UUID 'abcdef12-3456-7890-fedc-ba9876543210'. "
        "Classification is 'C', data mode 'REAL', frame type 'DARK' (for calibration, shhh!). This is super low priority, so 1500. "
        "The baseline autonomy RSO string is 'CATID-001,CATID-002,CATID-003'. Let this objective run until 2026-12-31 23:59:59+00:00. "
        "The RSO ID list should include 'RSO-X1', 'RSO-Y2', and 'RSO-Z3' for now, but you know it'll get updated!"
    ): {
        "objective_uuid": "abcdef12-3456-7890-fedc-ba9876543210",
        "classification_marking": "C",
        "data_mode": "REAL",
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "CATID-001,CATID-002,CATID-003",
        "objective_end_time": "datetime.datetime(2026, 12, 31, 23, 59, 59, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["RSO-X1", "RSO-Y2", "RSO-Z3"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 354: CatalogMaintenanceObjective
    (
        "Let's whip up a super-duper CatalogMaintenanceObjective! We need it for sensors 'ALPHA-SENSE-7' and 'OMEGA-EYE-9', tagged with a standard 'U' classification. "
        "Set the data mode to 'REAL' because this is the real deal, folks! Give it a shiny priority of 5 and a patience setting of 15 minutes so it doesn't wait around forever. "
        "Let's have this objective start its mission at 2025-06-15 10:00:00+00:00 and wrap things up by 2025-06-15 18:00:00+00:00. "
        "It should use 'RATE_TRACK_SIDEREAL' for tracking in the 'MEO' orbital playground. "
        "And don't forget the RSO ID hit list: '98765,43210,11223'. We also need to make sure the end time offset is a quick 10 minutes, and let's keep that visibility check switched off for this one!"
    ): {
        "binning": None,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 10,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 5,
        "rso_id_list": ["98765", "43210", "11223"],
        "sensor_name_list": ["ALPHA-SENSE-7", "OMEGA-EYE-9"],
        "visibility_check": False,
    },
    # New Example 355: SearchObjective
    (
        "Golly gee, we need a SearchObjective to find target 'TARGET-XYZ-1' using the amazing 'SUPER-SCOPE-SENSOR'. "
        "Mark it with a 'S' classification, operate in 'TEST' mode, and give it a high priority of 1! "
        "We'll use the 'RATE_TRACK_SIDEREAL' tracking method. "
        "The action starts at 2025-07-01 14:00:00+00:00 and finishes up at 2025-07-01 15:30:00+00:00. "
        "Set the initial offset to 90 seconds and the final offset to 120 seconds for a wide sweep! "
        "Frame overlap should be a solid 80%, and the end time offset should be 30 minutes. "
        "We need an 'ALONG_TRACK' search pattern, and the search itself should commence 20 minutes after the objective gets rolling. No visibility check needed for this test run!"
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 30,
        "final_offset": 120,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 7, 1, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "search_start_time": "datetime.datetime(2025, 7, 1, 14, 20, tzinfo=TzInfo(UTC))",  # 20 minutes after start
        "search_type": "ALONG_TRACK",
        "sensor_name": "SUPER-SCOPE-SENSOR",
        "target_id": "TARGET-XYZ-1",
        "visibility_check": False,
    },
    # New Example 356: GeodssRevisitObjective
    (
        "Alright team, let's set up a dazzling GeodssRevisitObjective! "
        "We're focusing on targets 'STAR-BRIGHT-01' and 'ORBIT-PAL-07' using sensors 'GEODSS-NORTH' and 'GEODSS-SOUTH'. "
        "Classification is 'U//FOUO', and we're running in 'REAL' data mode. "
        "The tracking type is 'RATE_TRACK_SIDEREAL', of course! "
        "The mission kicks off at 2025-08-20 20:00:00+00:00. "
        "Now for the fancy GEODSS knobs: set the readout rate to 1 (that's 2MHz!), gain setting to 0 (High Gain power!), SOI filter position to 2 (10% Light mode engaged!), auto track type to 2 (Manual control activated!), camera mode to 0 (Normal viewing!), array kind to 0 (Main array online!), binning mode to 0 (No Binning for maximum detail!), and scan mode to 0 (Continuous scanning!). "
        "Let's set patience to a generous 45 minutes and priority to 8."
    ): {
        "acquisition_type": 0,  # Default, not specified in prompt
        "array_kind": 0,
        "auto_track_roi_position": 0,  # Default, not specified in prompt
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 0,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default, not specified in prompt
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 0,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 1,  # Default, not specified in prompt
        "num_skip_frames": 0,  # Default, not specified in prompt
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 20, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default, not specified in prompt
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default, not specified in prompt
        "patience_minutes": 45,
        "priority": 8,
        "rate_track_verify": 0,  # Default, not specified in prompt
        "readout_rate_setting": 1,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEODSS-NORTH", "GEODSS-SOUTH"],
        "soi_filter_position": 2,
        "target_id_list": ["STAR-BRIGHT-01", "ORBIT-PAL-07"],
        "visibility_check": False,  # Default
    },
    # New Example 357: PeriodicRevisitObjective
    (
        "Wowza! Let's schedule a PeriodicRevisitObjective for targets 'COSMIC-FRIEND-3', 'SPACE-BUDDY-5', and 'ASTRO-CHUM-8'. "
        "Use sensors 'SENSORIUM-PRIME' and 'OPTIC-FANTASTIC'. "
        "This needs an 'C' classification and should run in 'EXERCISE' data mode. "
        "Give it a priority of 15 and a patience of 60 minutes. "
        "We'll kick this off at 2025-09-05 05:00:00+00:00. "
        "We want to capture 10 frames per intent, each with an integration time of 3.5 seconds. "
        "Don't ignore other objective submissions; we want to be good neighbors! "
    ): {
        "classification_marking": "C",
        "target_id_list": ["COSMIC-FRIEND-3", "SPACE-BUDDY-5", "ASTRO-CHUM-8"],
        "sensor_name_list": ["SENSORIUM-PRIME", "OPTIC-FANTASTIC"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 60,
        "revisits_per_hour": None,
        "number_of_frames": 10,
        "integration_time": 3.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 5, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 400,  # Default
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 358: UctObservationObjective
    (
        "Holy cats! We've spotted some Uncorrelated Targets! Let's create a UctObservationObjective to keep an eye on UCT RSOs 'UCT-ALPHA', 'UCT-BETA', and 'UCT-GAMMA'. "
        "We'll use sensors 'SPACE-TELESCOPE-X' and 'ORBITAL-CAMERA-Y'. "
        "This mission is classified 'TS', running in 'REAL' mode in the 'GEO' orbital belt. "
        "We need to get 10 revisits per hour, stat! "
        "Start this objective right away at 2025-10-10 22:00:00+00:00. "
        "Visibility check is absolutely true for these elusive targets! "
        "Let's also sort by the brightest UCTs to prioritize those. "
        "Set the end time offset to a snappy 90 minutes, capture 3 frames per intent, each for 5 seconds."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["UCT-ALPHA", "UCT-BETA", "UCT-GAMMA"],
        "sensor_name_list": ["SPACE-TELESCOPE-X", "ORBITAL-CAMERA-Y"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 10.0,
        "number_of_frames": 3,
        "integration_time": 5.0,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 22, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 10,  # Default
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 359: SingleIntentObjective
    (
        "Booyah! We need a SingleIntentObjective for target ID 'CRITICAL-TARGET-Z' and RSO ID 'SPECIAL-RSO-A'. "
        "We'll assign sensors 'NINJA-SENSOR-1' and 'STEALTH-CAM-2' to this mission. "
        "It's a 'C' classification, running in 'SIMULATED' data mode. "
        "Use 'RATE_TRACK' for the tracking type and 'LIGHT' for the frame type. "
        "This is important, so give it a priority of 3! "
        "Start this objective at 2025-11-11 11:11:00+00:00. "
        "We need 8 frames with an integration time of 1.5 seconds, and set the binning to 1."
    ): {
        "classification_marking": "C",
        "target_id": "CRITICAL-TARGET-Z",
        "rso_id": "SPECIAL-RSO-A",
        "sensor_name_list": ["NINJA-SENSOR-1", "STEALTH-CAM-2"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "number_of_frames": 8,
        "integration_time": 1.5,
        "priority": 3,
        "binning": 1,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 11, 11, 11, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 360: DataEnrichmentObjective
    (
        "Zoinks! Time for a DataEnrichmentObjective to supercharge our info on targets 'INFO-BOOST-4', 'DATA-GOLDMINE-6', and 'KNOWLEDGE-NODE-9'. "
        "Sensors 'COLLECTOR-UNIT-A' and 'PROCESSOR-ARRAY-B' are on the job! "
        "This objective has a 'U//FOUO' classification and runs in 'REAL' data mode. "
        "Let's set the maximum number of RSOs to observe at a time to a cool 10, and aim for 15 revisits per hour! "
        "Start this information quest at 2026-01-01 01:00:00+00:00. "
        "Visibility check is a must, so make it true! Priority can be the default 20."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["INFO-BOOST-4", "DATA-GOLDMINE-6", "KNOWLEDGE-NODE-9"],
        "sensor_name_list": ["COLLECTOR-UNIT-A", "PROCESSOR-ARRAY-B"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 1, 1, 1, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 20,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 361: SensorCheckoutObjective
    (
        "It's time for a tune-up! Let's get a SensorCheckoutObjective ready for sensor 'MAINTENANCE-BOT-1'. "
        "Give it a 'U' classification and run it in 'REAL' data mode. "
        "This checkout will focus on the 'GEO' orbital regime using 'RATE_TRACK_SIDEREAL' tracking. "
        "Start the process at 2026-02-14 10:00:00+00:00. "
        "Set revisits per hour to 2.5, patience to 40 minutes, and make sure the visibility check is enabled (true!). "
        "We'll need 7 frames per intent, each with a 2.8-second integration time."
    ): {
        "classification_marking": "U",
        "sensor_name": "MAINTENANCE-BOT-1",
        "orbital_regime": "GEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.5,
        "number_of_frames": 7,
        "integration_time": 2.8,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 14, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 10,  # Default
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 362: BaselineAutonomyObjective
    (
        "Alright team, time for a continuous watch! Initiate a BaselineAutonomyObjective with UUID 'f917d4a2-5b1c-4e8f-8c0a-3d2e1b4f6c5d'. "
        "It should use 'U' markings, operate in 'REAL' data mode, and capture 'LIGHT' frames. "
        "Assign it a super-low priority of 1500 so it doesn't hog resources. "
        "The baseline autonomy will track catalog IDs '54321' and '98765'. "
        "Include RSO IDs 'PLANET-PAL-1', 'ORBIT-WATCHER-2', and 'STELLAR-GUARD-3' in the initial list. "
        "Crucially, this objective should have no end time, running continuously until stopped manually!"
    ): {
        "objective_uuid": "f917d4a2-5b1c-4e8f-8c0a-3d2e1b4f6c5d",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1500,
        "baseline_autonomy_rso": "54321,98765",
        "objective_end_time": None,
        "rso_id_list": ["PLANET-PAL-1", "ORBIT-WATCHER-2", "STELLAR-GUARD-3"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 363: CatalogMaintenanceObjective
    (
        "Say, Doc! I need a CatalogMaintenanceObjective for those fancy sensors RME11 and LMNT09 with S classification marking. "
        "Make it REAL mode with super duper high priority 5! Set patience to a whopping 45 mins and end time offset of 30 mins. "
        "Start the whole shebang at 2025-06-15 08:45:00+00:00 and end it at 2025-06-15 14:30:00+00:00. "
        "Oh, and use RATE_TRACK_SIDEREAL tracking in MEO regime with visibility check true. RSO ID list should have '33221,44556,77889'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 15, 14, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 15, 8, 45, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 5,
        "rso_id_list": ["33221", "44556", "77889"],
        "sensor_name_list": ["RME11", "LMNT09"],
        "visibility_check": True,
    },
    # New Example 364: SearchObjective
    (
        "What's up, Doc? I need a SearchObjective for target 55443 with sensor LMNT22. Use TS classification marking and SIMULATED mode. "
        "Set priority to 8 and use RATE_TRACK tracking. The whole operation should start at 2025-07-13 15:20:00+00:00 and finish by 2025-07-13 17:00:00+00:00. "
        "For the search params, I need initial offset of 120 seconds, final offset of 180 seconds, and a whopping 80% frame overlap! "
        "End time offset should be 35 minutes. Use CROSS_TRACK search type with search start time 20 minutes after objective start."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 35,
        "final_offset": 180,
        "frame_overlap_percentage": 0.8,
        "frame_type": "LIGHT",
        "initial_offset": 120,
        "integration_time": None,
        "number_of_frames": None,
        "objective_end_time": "datetime.datetime(2025, 7, 13, 17, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 13, 15, 20, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 8,
        "search_start_time": "datetime.datetime(2025, 7, 13, 15, 40, tzinfo=TzInfo(UTC))",
        "search_type": "CROSS_TRACK",
        "sensor_name": "LMNT22",
        "target_id": "55443",
        "visibility_check": False,
    },
    # New Example 365: GeodssRevisitObjective
    (
        "Eh, what's up, Doc? I'm huntin' for a GeodssRevisitObjective for targets 77889,11223 with sensors RME07,LMNT08. "
        "Set it to U classification, TEST mode, and RATE_TRACK_SIDEREAL trackin'. Make the priority 15 and start at 2025-09-05 22:10:00+00:00. "
        "Configure those fancy GEODSS settings with readout_rate 0 (1MHz), gain_setting 1 (Low Gain), soi_filter 2 (10% Light), "
        "auto_track_type 2 (Manual), camera_mode 1 (Zoomed EBS), array_kind 1 (Photometer), binning_mode 0 (No Binning), scan_mode 0 (Continuous)."
    ): {
        "acquisition_type": 0,
        "array_kind": 1,
        "auto_track_roi_position": 0,
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,
        "data_mode": "TEST",
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
        "objective_start_time": "datetime.datetime(2025, 9, 5, 22, 10, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0,
        "optimal_frames_per_hour": 400,
        "overscan": 0,
        "patience_minutes": 30,
        "priority": 15,
        "rate_track_verify": 0,
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["RME07", "LMNT08"],
        "soi_filter_position": 2,
        "target_id_list": ["77889", "11223"],
        "visibility_check": False,
    },
    # New Example 366: PeriodicRevisitObjective
    (
        "I say, I say, I need a PeriodicRevisitObjective for those confounded targets 98765,54321 using sensors RME03,LMNT04. "
        "Give it a U//FOUO marking and EXERCISE mode. Set priority to 7 with patience minutes of 25. "
        "Make it ignore other objective submissions true and optimal frames per hour 500. Start this contraption at 2025-10-17 03:15:00+00:00. "
        "Let's have 3 revisits per hour with 8 frames per intent and 1.5 second integration time."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["98765", "54321"],
        "sensor_name_list": ["RME03", "LMNT04"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 25,
        "revisits_per_hour": 3.0,
        "number_of_frames": 8,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 10, 17, 3, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 367: UctObservationObjective
    (
        "Sufferin' succotash! Create a UctObservationObjective for the UCT RSOs 45678,91011 using sensors RME25,LMNT27. "
        "Use C classification marking, REAL mode, and set it for the MEO orbital regime. Priority should be 4, with 8 revisits per hour. "
        "Begin at 2025-11-30 14:40:00+00:00 and set end time offset to 90 minutes. Turn on visibility check and make patience 40 minutes. "
        "Configure for 6 frames with 3 second integration time and sorting by brightest UCT enabled."
    ): {
        "classification_marking": "C",
        "uct_rso_id_list": ["45678", "91011"],
        "sensor_name_list": ["RME25", "LMNT27"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "orbital_regime": "MEO",
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 8.0,
        "number_of_frames": 6,
        "integration_time": 3,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 30, 14, 40, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 4,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 368: SingleIntentObjective
    (
        "Great horny toads! I require a SingleIntentObjective with target ID 12321 and RSO ID 56789 using sensors RME19,LMNT20. "
        "Make it S classification and SIMULATED mode with SIDEREAL tracking. Set priority to 6. "
        "Start this rascally objective at 2025-12-25 00:01:00+00:00. Configure it for 10 frames with 2.5 second integration time and binning 4."
    ): {
        "classification_marking": "S",
        "target_id": "12321",
        "rso_id": "56789",
        "sensor_name_list": ["RME19", "LMNT20"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "number_of_frames": 10,
        "integration_time": 2.5,
        "priority": 6,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 25, 0, 1, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 369: DataEnrichmentObjective
    (
        "You're despicable! But I need a DataEnrichmentObjective for targets 11111,22222,33333 using sensors RME42,LMNT44. "
        "Configure with TS classification marking and EXERCISE mode. Set collect_request_type to RATE_TRACK, max RSO to observe as 12, and 15 revisits per hour. "
        "Start this despicable plan at 2026-02-14 13:37:00+00:00. Ensure visibility check is enabled and priority is set to 3."
    ): {
        "classification_marking": "TS",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["11111", "22222", "33333"],
        "sensor_name_list": ["RME42", "LMNT44"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",
        "binning": None,
        "max_rso_to_observe": 12,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 2, 14, 13, 37, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 3,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 370: SensorCheckoutObjective
    (
        "That's all folks! Just kidding, I need a SensorCheckoutObjective with classification_marking='C' and sensor_name='LMNT51'. "
        "Configure it for XGEO orbital regime, TEST mode, and RATE_TRACK_SIDEREAL collect type. Set priority to 9, with 2.5 revisits per hour. "
        "Start at 2026-04-01 04:20:00+00:00 with visibility check enabled. Set patience to 35 minutes, with 7 frames and 1.75 second integration time."
    ): {
        "classification_marking": "C",
        "sensor_name": "LMNT51",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": True,
        "patience_minutes": 35,
        "revisits_per_hour": 2.5,
        "number_of_frames": 7,
        "integration_time": 1.75,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 1, 4, 20, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 9,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 371: BaselineAutonomyObjective
    (
        "I'm hunting wabbits... I mean objectives! Create a BaselineAutonomyObjective with UUID '987f6543-c21b-34e5-d678-987654321000'. "
        "Use S//FOUO classification, EXERCISE mode, and DARK frame type with priority 800. Add RSO IDs 54321, 98765, and 13579, "
        "along with catalog IDs 27890 and 31415 for continuous tracking. This objective should run indefinitely with no end time."
    ): {
        "objective_uuid": "987f6543-c21b-34e5-d678-987654321000",
        "classification_marking": "S//FOUO",
        "data_mode": "EXERCISE",
        "frame_type": "DARK",
        "priority": 800,
        "baseline_autonomy_rso": "27890,31415",
        "objective_end_time": None,
        "rso_id_list": ["54321", "98765", "13579"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 372: CatalogMaintenanceObjective
    (
        "Alright, ya varmint! Create a CatalogMaintenanceObjective, see? We gotta keep tabs on those pesky space-critters, RSO IDs 'BUGZ86' and 'DAFFY99', with U//FOUO markings, ya hear? "
        "This ain't no tea party, so set the data_mode to REAL. I want a priority of, let's say, 55. Give me a patience of 45 whole minutes, and an end time offset of 30 minutes. And make sure that visibility check is set to true, I don't want any surprises! "
        "We'll kick this shindig off at 2025-10-31 00:00:00+00:00, and wrap it up by 2025-10-31 06:00:00+00:00. We're trackin' 'em in the MEO regime, with RATE_TRACK. Oh, and the sensors? Use 'ELMER01' and 'YOSEMITE02'. That'll show 'em!"
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "end_time_offset_minutes": 30,
        "priority": 55,
        "sensor_name_list": ["ELMER01", "YOSEMITE02"],
        "rso_id_list": ["BUGZ86", "DAFFY99"],
        "objective_start_time": "datetime.datetime(2025, 10, 31, 0, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 10, 31, 6, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "frame_type": "LIGHT",
        "binning": None,
        "visibility_check": True,
        "objective_name": "CatalogMaintenanceObjective",
    },
    # New Example 373: SearchObjective
    (
        "Listen up, see? I'm lookin' for me gold! Create a SearchObjective for target ID 'GOLDMINE49' using sensor 'SAM007'. Mark it S for Secret, and make it REAL mode. This is top priority, so set it to 1, ya galoot! "
        "Use SIDEREAL tracking. Start this here search at 2025-11-15 10:00:00+00:00 and don't stop 'til 2025-11-15 14:30:00+00:00, or I'll be hornswoggled! "
        "I need an initial offset of 120 seconds, and a final offset of 150 seconds. Make sure the frame overlap is a good 60 percent, not a smidgen less! The end time offset should be 50 minutes. "
        "The search type is, uh, RASTER_SCAN, and the search better start 20 minutes after the objective kicks off. Don't forget, number of frames is 10 and integration time is 0.5 seconds, see?"
    ): {
        "classification_marking": "S",
        "target_id": "GOLDMINE49",
        "sensor_name": "SAM007",
        "search_type": "RASTER_SCAN",  # Assuming RASTER_SCAN is a valid SearchType
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,  # Default, but being explicit
        "initial_offset": 120,
        "final_offset": 150,
        "frame_overlap_percentage": 0.6,
        "end_time_offset_minutes": 50,
        "priority": 1,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 15, 10, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 11, 15, 14, 30, 0, tzinfo=TzInfo(UTC))",
        "number_of_frames": 10,
        "integration_time": 0.5,
        "search_start_time": "datetime.datetime(2025, 11, 15, 10, 20, 0, tzinfo=TzInfo(UTC))",  # objective_start_time + 20 minutes
        "objective_name": "SearchObjective",
    },
    # New Example 374: GeodssRevisitObjective
    (
        "Oh, dear! Prepare a GeodssRevisitObjective. It's for my... research on Earth (it obstructs my view of Venus, you see). Target IDs are 'PLANETX' and 'ASTEROID_Q36'. Use sensors 'K9UNIT_A' and 'K9UNIT_B'. "
        "This requires a U marking, naturally, and REAL mode. Priority is a modest 7. Tracking is RATE_TRACK_SIDEREAL. "
        "Commence operations at 2025-12-01 03:00:00+00:00. We'll need the readout_rate at 0 (that's 1MHz, isn't it marvelous?), gain_setting at 1 (Low Gain, for subtlety), and the soi_filter at 2 (10% Light). "
        "Set auto_track_type to 2 (Manual, I have very specific requirements), camera_mode to 1 (Zoomed EBS, for close-ups!), array_kind to 0 (Main, of course), binning_mode to 0 (No Binning), and scan_mode to 0 (Continuous, darling, continuous!). Oh, and 5 revisits per hour should suffice."
    ): {
        "classification_marking": "U",
        "target_id_list": ["PLANETX", "ASTEROID_Q36"],
        "sensor_name_list": ["K9UNIT_A", "K9UNIT_B"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",
        "visibility_check": False,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 5.0,
        "number_of_frames": None,
        "integration_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 12, 1, 3, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": False,
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
        "array_kind": 0,
        "binning_mode": 0,
        "scan_mode": 0,
        "overscan": 0,  # Default
        "command": 0,  # Default
        "objective_name": "GeodssRevisitObjective",
    },
    # New Example 375: PeriodicRevisitObjective
    (
        "I say, I say, son! We need a PeriodicRevisitObjective! For targets, uh, 'SHUTTLECOCK_A1' and 'FEATHER_D2'. The sensors? Why, 'FOGHORN_CAM1' and 'LEGHORN_SCOPE2', of course! "
        "Mark this 'C' for Classified, it's a high-stakes tournament, you know. TEST mode for now, just practicing. Priority's a solid 3. Patience of, say, 20 minutes. And don't you go ignoring other objectives, boy! "
        "Start this shebang at 2026-01-10 14:00:00+00:00. I want, I say, I want 10 revisits per hour! And give me 3 frames per intent, with an integration time of 1.5 seconds. Optimal frames per hour? 300 sounds about right, son!"
    ): {
        "classification_marking": "C",
        "target_id_list": ["SHUTTLECOCK_A1", "FEATHER_D2"],
        "sensor_name_list": ["FOGHORN_CAM1", "LEGHORN_SCOPE2"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": False,  # Default
        "patience_minutes": 20,
        "revisits_per_hour": 10.0,
        "number_of_frames": 3,
        "integration_time": 1.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 1, 10, 14, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 3,
        "ignore_other_objective_intent_submissions": False,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 376: UctObservationObjective
    (
        "It's mine! All mine! Create a UctObservationObjective for my shiny UCT RSOs: 'QUACKERS_JUNK1' and 'SPACE_DUCKLING7'. Use sensors 'DUCKDODGERS_EYE' and 'CADET_CAM'. "
        "Marking is U//FOUO, it's top-secret duck business! REAL mode, GEO regime for those high-flying thingamajigs. Priority is 15, not too shabby. I want 8 magnificent revisits per hour! "
        "Get this party started at 2026-02-20 09:30:00+00:00. And you better sort by the brightest UCT, I only want the best! End time offset is a generous 75 minutes. Visibility check must be true, no hiding from Duck Dodgers! "
        "Oh, and 7 frames, 2.5 seconds integration time. Make it snappy!"
    ): {
        "classification_marking": "U//FOUO",
        "uct_rso_id_list": ["QUACKERS_JUNK1", "SPACE_DUCKLING7"],
        "sensor_name_list": ["DUCKDODGERS_EYE", "CADET_CAM"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 2.5,
        "binning": None,
        "end_time_offset_minutes": 75,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 9, 30, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 15,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 377: SingleIntentObjective
    (
        "Meep meep! Oops, force of habit. Erm, construct a SingleIntentObjective. Target ID is 'ROADRUNNER_X1', RSO ID is 'ACME_RR_TRACKER'. Sensors are 'COYOTE_SCOPE_9000' and 'WILE_E_CAM_PRO'. "
        "Marking: U. Mode: REAL. Tracking: RATE_TRACK. Priority: 1 (this HAS to work!). "
        "Objective starts precisely at 2026-03-05 12:00:00+00:00. I need 2 frames, an integration time of 0.1 seconds for super-speed, and binning of 4. This is my magnum opus!"
    ): {
        "classification_marking": "U",
        "target_id": "ROADRUNNER_X1",
        "rso_id": "ACME_RR_TRACKER",
        "sensor_name_list": ["COYOTE_SCOPE_9000", "WILE_E_CAM_PRO"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 2,
        "integration_time": 0.1,
        "priority": 1,
        "binning": 4,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 5, 12, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # This objective is for a single shot, effectively.
        "objective_name": "SingleIntentObjective",
    },
    # New Example 378: DataEnrichmentObjective
    (
        "TAZ WANT DATA! MAKE DataEnrichmentObjective! For targets 'SPINNY_THING_1', 'LOOKS_LIKE_CHICKEN_DRUMSTICK', 'FAST_ROCKS'. Sensors 'TAZ_EYE_LEFT', 'TAZ_EYE_RIGHT'. "
        "Marking S for 'Secret Taz Stash'. REAL data mode, because TAZ NO LIKE FAKE! Use RATE_TRACK_SIDEREAL. "
        "Max RSO to observe: 5, TAZ GET DIZZY WITH MORE. 15 revisits per hour, TAZ HUNGRY FOR MANY LOOKS! "
        "Start when TAZ SAY START! 2026-04-18 15:45:00+00:00. Visibility check, YES, TAZ WANT TO SEE! Priority? 25, TAZ PATIENT...SOMETIMES."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": [
            "SPINNY_THING_1",
            "LOOKS_LIKE_CHICKEN_DRUMSTICK",
            "FAST_ROCKS",
        ],
        "sensor_name_list": ["TAZ_EYE_LEFT", "TAZ_EYE_RIGHT"],
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 5,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 4, 18, 15, 45, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 379: SensorCheckoutObjective
    (
        "I tawt I taw a puddy tat! Er, I mean, create a SensorCheckoutObjective. Classification marking is 'U'. The sensor in question is 'TWEETY_SPOTTER_5X'. "
        "Data mode is REAL, orbital regime is XGEO (that wily bird gets everywhere!). Collect request type is RATE_TRACK. Priority 5, this is impo'tant. "
        "I need 0.5 revisits per hour, just a quick peek. Start this checkout on 2026-05-01 08:00:00+00:00. Visibility must be checked, of course. "
        "Patience can be 15 minutes. Give me 4 frames, and an integration time of 3 seconds. We gotta make sure this thing works, see?"
    ): {
        "classification_marking": "U",
        "sensor_name": "TWEETY_SPOTTER_5X",
        "orbital_regime": "XGEO",
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 15,
        "revisits_per_hour": 0.5,
        "number_of_frames": 4,
        "integration_time": 3.0,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 5, 1, 8, 0, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 5,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 380: BaselineAutonomyObjective
    (
        "Th-th-th-that's the plan! Let's set up a BaselineAutonomyObjective. The UUID for this one, by golly, is 'PORKY-PIG-NETWORK-001'. "
        "Markings are U, for Unremarkable... I mean, Unclassified! Data mode: REAL. Frame type: LIGHT. This is a low, low priority, so make it 1500. "
        "The baseline autonomy RSO catalog IDs are 'BORING_ROCK_101,OLD_SATELLITE_202,SPACE_DUST_303'. "
        "We want this to run indefinitely, so no end time, please! The RSO ID list can be 'TBD_RSO_1,TBD_RSO_2' for now, the system will figure it out. Th-th-th-that's all!"
    ): {
        "objective_uuid": "PORKY-PIG-NETWORK-001",
        "classification_marking": "U",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 1500,
        "baseline_autonomy_rso": "BORING_ROCK_101,OLD_SATELLITE_202,SPACE_DUST_303",
        "objective_end_time": None,
        "rso_id_list": ["TBD_RSO_1", "TBD_RSO_2"],
        "objective_name": "BaselineAutonomyObjective",
    },
    # New Example 381: CatalogMaintenanceObjective
    (
        "Yeehaw! Let's rustle up a CatalogMaintenanceObjective for those trusty sensors "
        "RME08 and LMNT11. We'll slap a U markings on this here request, run it in TEST mode, "
        "and give it a humdinger of a priority, say, 15. Don't you fret none about patience, "
        "we've got a solid 20 minutes to spare! We'll set the end time offset to a cool 30 minutes, "
        "and let's skip that visibility check, we're feelin' lucky! "
        "Kick this whole shebang off at 2025-06-01 10:00:00+00:00 and wrap it up by "
        "2025-06-01 14:00:00+00:00. We'll use RATE_TRACK_SIDEREAL for tracking, and it's all about "
        "the MEO regime this time, folks! The RSO ID list? Oh, you betcha! It's '98765,43210'."
    ): {
        "binning": None,
        "classification_marking": "U",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "TEST",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",  # Default value from schema
        "objective_end_time": "datetime.datetime(2025, 6, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",  # Default name
        "objective_start_time": "datetime.datetime(2025, 6, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,  # Default value from schema
        "orbital_regime": "MEO",
        "patience_minutes": 20,
        "priority": 15,
        "rso_id_list": ["98765", "43210"],
        "sensor_name_list": ["RME08", "LMNT11"],
        "visibility_check": False,
    },
    # New Example 382: SearchObjective
    (
        "By my beard! We need a SearchObjective pronto for target 99887, using sensor "
        "WHT05! Let's give it that shiny S marking, and we'll run it in REAL mode, "
        "because, you know, it's *real* important! Priority? A respectable 8, if I do say so myself. "
        "We'll stick with the ol' RATE_TRACK_SIDEREAL for tracking. "
        "Start this cosmic scavenger hunt at 2025-06-05 15:30:00+00:00 and finish up at "
        "2025-06-05 17:00:00+00:00. Set the initial offset to 120 seconds, final offset to 180 seconds, "
        "and let's get a frame overlap of 85%! The end time offset shall be 60 minutes. "
        "We're going with an ACROSS_TRACK search type, and the search itself should begin "
        "a full 20 minutes after the objective starts!"
    ): {
        "binning": None,  # Default value from schema
        "classification_marking": "S",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 60,
        "final_offset": 180,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",  # Default value from schema
        "initial_offset": 120,
        "integration_time": None,  # Default value from schema
        "number_of_frames": None,  # Default value from schema
        "objective_end_time": "datetime.datetime(2025, 6, 5, 17, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",  # Default name
        "objective_start_time": "datetime.datetime(2025, 6, 5, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,  # Default value from schema
        "priority": 8,
        "search_start_time": "datetime.datetime(2025, 6, 5, 15, 50, tzinfo=TzInfo(UTC))",  # Calculated from prompt
        "search_type": "ACROSS_TRACK",
        "sensor_name": "WHT05",
        "target_id": "99887",
        "visibility_check": False,  # Default value from schema
    },
    # New Example 383: GeodssRevisitObjective
    (
        "Listen up, space cadets! We need a GeodssRevisitObjective for targets "
        "10101, 20202, using sensors RME16, LMNT18! Mark it U//FOUO, because, well, it's for official "
        "use only, naturally! We're running in REAL mode with RATE_TRACK_SIDEREAL tracking. "
        "Priority? A solid 7! Start this cosmic ballet at 2025-06-10 20:00:00+00:00. "
        "Let's set the readout rate to 0 (that's 1MHz!), the gain setting to 1 (Low Gain!), "
        "soi filter position to 2 (10% Light!), auto track type to 2 (Manual!), camera mode to 1 (Zoomed EBS!), "
        "array kind to 1 (Photometer!), binning mode to 0 (No Binning!), and scan mode to 0 (Continuous!). "
        "Don't even think about ignoring other objectives' intent submissions on this one, no sirree!"
    ): {
        "acquisition_type": 0,  # Default value from schema
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default value from schema
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default value from schema
        "data_mode": "REAL",
        "frame_type": "LIGHT",  # Default value from schema
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,  # Default value from schema
        "intent_end_time": None,  # Default value from schema
        "intent_start_time": None,  # Default value from schema
        "num_observations": 1,  # Default value from schema
        "num_skip_frames": 0,  # Default value from schema
        "number_of_frames": None,  # Default value from schema
        "objective_end_time": None,  # Default value from schema
        "objective_name": "GeodssRevisitObjective",  # Default name
        "objective_start_time": "datetime.datetime(2025, 6, 10, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,  # Default value from schema
        "observation_interval": 0.0,  # Default value from schema
        "optimal_frames_per_hour": 400,  # Default value from schema
        "overscan": 0,  # Default value from schema
        "patience_minutes": 30,  # Default value from schema
        "priority": 7,
        "rate_track_verify": 0,  # Default value from schema
        "readout_rate_setting": 0,
        "revisits_per_hour": None,  # Default value from schema
        "scan_mode": 0,
        "sensor_name_list": ["RME16", "LMNT18"],
        "soi_filter_position": 2,
        "target_id_list": ["10101", "20202"],
        "visibility_check": False,  # Default value from schema
    },
    # New Example 384: PeriodicRevisitObjective
    (
        "Alright, let's cook up a PeriodicRevisitObjective! We're targeting IDs 33445 and 77889 "
        "with sensors RME09 and LMNT13. This one gets a C marking, runs in SIMULATED mode, "
        "and has a priority of a nice round 5. We'll give it a generous 45 minutes of patience, "
        "and we absolutely WILL ignore other objective submissions this time! "
        "Start this escapade at 2025-06-15 14:00:00+00:00. "
        "We want an optimal frames per hour of 500, number of frames set to 10, and an "
        "integration time of 3.5 seconds. This is going to be a doozy!"
    ): {
        "classification_marking": "C",
        "target_id_list": ["33445", "77889"],
        "sensor_name_list": ["RME09", "LMNT13"],
        "data_mode": "SIMULATED",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default value from schema
        "frame_type": "LIGHT",  # Default value from schema
        "visibility_check": False,  # Default value from schema
        "patience_minutes": 45,
        "revisits_per_hour": None,  # Default value from schema
        "number_of_frames": 10,
        "integration_time": 3.5,
        "binning": None,  # Default value from schema
        "objective_start_time": "datetime.datetime(2025, 6, 15, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default value from schema
        "priority": 5,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 500,
        "objective_uuid": None,  # Default value from schema
        "intent_start_time": None,  # Default value from schema
        "intent_end_time": None,  # Default value from schema
        "objective_name": "PeriodicRevisitObjective",  # Default name
    },
    # New Example 385: UctObservationObjective
    (
        "Great horny toads! We need a UctObservationObjective for UCT RSOs 11223 and 44556 "
        "using sensors RME20 and LMNT21. This one's classified S, runs in REAL mode, and is focused on "
        "the GEO regime. Priority? A super-duper 1! We want a snappy 10 revisits per hour! "
        "Let's get this show on the road at 2025-06-20 21:00:00+00:00. Make sure visibility check is TRUE, "
        "set the end time offset to a solid 90 minutes, and definitely sort by the brightest UCTs! "
        "Oh, and set the number of frames to 8 and integration time to 1.5 seconds. Make it snappy!"
    ): {
        "classification_marking": "S",
        "uct_rso_id_list": ["11223", "44556"],
        "sensor_name_list": ["RME20", "LMNT21"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default value from schema
        "frame_type": "LIGHT",  # Default value from schema
        "orbital_regime": "GEO",
        "visibility_check": True,
        "patience_minutes": 30,  # Default value from schema
        "revisits_per_hour": 10.0,
        "number_of_frames": 8,
        "integration_time": 1.5,
        "binning": None,  # Default value from schema
        "end_time_offset_minutes": 90,
        "objective_uuid": None,  # Default value from schema
        "objective_start_time": "datetime.datetime(2025, 6, 20, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default value from schema
        "priority": 1,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",  # Default name
    },
    # New Example 386: SingleIntentObjective
    (
        "What's up, Doc? We need a SingleIntentObjective! The target ID is 55667, and the RSO ID is 22334. "
        "We'll use sensors RME23 and LMNT25 for this caper. It's a TS marking, running in EXERCISE mode, "
        "using SIDEREAL tracking. Priority? A whopping 3! "
        "Let's get this objective going at 2025-06-25 09:00:00+00:00. "
        "We need 12 frames, an integration time of 0.5 seconds, and binning set to 4. "
        "This is a one-shot deal, so no need for end times or anything fancy!"
    ): {
        "classification_marking": "TS",
        "target_id": "55667",
        "rso_id": "22334",
        "sensor_name_list": ["RME23", "LMNT25"],
        "data_mode": "EXERCISE",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default value from schema
        "number_of_frames": 12,
        "integration_time": 0.5,
        "priority": 3,
        "binning": 4,
        "intent_start_time": None,  # Default value from schema
        "intent_end_time": None,  # Default value from schema
        "objective_uuid": None,  # Default value from schema
        "objective_start_time": "datetime.datetime(2025, 6, 25, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default value from schema
        "objective_name": "SingleIntentObjective",  # Default name
    },
    # New Example 387: DataEnrichmentObjective
    (
        "Holy high heavens! It's time for a DataEnrichmentObjective! We're lookin' at targets "
        "88776, 11223, and 44556, using sensors RME33, LMNT36. This one's got a C marking, runs "
        "in REAL mode, and uses good old RATE_TRACK tracking! "
        "We wanna observe a maximum of 10 RSOs and get 15 revisits per hour. "
        "Start this data party at 2025-07-01 13:30:00+00:00. "
        "Visibility check? You bet your boots, set that to TRUE!"
    ): {
        "classification_marking": "C",
        "data_mode": "REAL",
        "objective_uuid": None,  # Default value from schema
        "target_id_list": ["88776", "11223", "44556"],
        "sensor_name_list": ["RME33", "LMNT36"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default value from schema
        "binning": None,  # Default value from schema
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2025, 7, 1, 13, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default value from schema
        "priority": 20,  # Default value from schema
        "intent_start_time": None,  # Default value from schema
        "intent_end_time": None,  # Default value from schema
        "visibility_check": True,
        "objective_name": "DataEnrichmentObjective",  # Default name
    },
    # New Example 388: SensorCheckoutObjective
    (
        "Alright, time for a check-up! Create a SensorCheckoutObjective with a classification_marking of 'U//FOUO' "
        "and focus on sensor 'UKR07'. We need this running in TEST mode, targeting the XGEO regime, "
        "using SIDEREAL tracking. Priority? A nice low 15. "
        "We're aiming for 2 revisits per hour, starting this objective on "
        "2025-07-10 18:00:00+00:00. Visibility check is a go (true!), give it 40 minutes of patience, "
        "and set the number of frames to 7 with an integration time of 3 seconds. Let's see if that sensor's tickin'!"
    ): {
        "classification_marking": "U//FOUO",
        "sensor_name": "UKR07",
        "orbital_regime": "XGEO",
        "data_mode": "TEST",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default value from schema
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 2.0,
        "number_of_frames": 7,
        "integration_time": 3.0,
        "binning": None,  # Default value from schema
        "objective_uuid": None,  # Default value from schema
        "objective_start_time": "datetime.datetime(2025, 7, 10, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,  # Default value from schema
        "intent_start_time": None,  # Default value from schema
        "intent_end_time": None,  # Default value from schema
        "priority": 15,
        "objective_name": "SensorCheckoutObjective",  # Default name
    },
    # New Example 389: BaselineAutonomyObjective
    (
        "Listen here, sister! We need a BaselineAutonomyObjective with a snazzy UUID: '98765432-10fe-56d4-b321-abcdef123456'. "
        "This baby gets C markings, runs in REAL mode, LIGHT frame type, and a rock-bottom priority of 2000! "
        "It's going to use RSO IDs: 13579, 24680, and 97531. And for the catalog IDs? "
        "We're talking '21234' and '25262'. This objective should run continuously, forever and ever, amen! (Meaning no end time!)"
    ): {
        "objective_uuid": "98765432-10fe-56d4-b321-abcdef123456",
        "classification_marking": "C",
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "priority": 2000,
        "baseline_autonomy_rso": "21234,25262",
        "objective_end_time": None,
        "rso_id_list": ["13579", "24680", "97531"],
        "objective_name": "BaselineAutonomyObjective",  # Default name
    },
    # New Example 390: CatalogMaintenanceObjective
    (
        "Generate a CatalogMaintenanceObjective. Utilize sensors SKYNET01 and HAL9000. The classification marking is S. "
        "Engage REAL data mode. Set priority to 50. Patience is 15 minutes. "
        "The end time offset is 30 minutes. Visibility check should be true. "
        "Objective initiates on 2025-11-01 10:00:00+00:00 and terminates on 2025-11-01 15:30:00+00:00. "
        "Employ SIDEREAL tracking. The orbital regime is MEO. "
        "The RSO ID list contains '55001', '55002', and '55003'. Specify binning as 2."
    ): {
        "binning": 2,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 11, 1, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 11, 1, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 15,
        "priority": 50,
        "rso_id_list": ["55001", "55002", "55003"],
        "sensor_name_list": ["SKYNET01", "HAL9000"],
        "visibility_check": True,
    },
    # New Example 391: SearchObjective
    (
        "Define a SearchObjective for target ID 'alpha-007' with sensor 'WALLY7'. "
        "Marking is C. Data mode is SIMULATED. Priority is exceptionally high at 1. "
        "Use RATE_TRACK tracking. The objective begins at 2025-12-15 08:00:00+00:00 and concludes at 2025-12-15 10:00:00+00:00. "
        "Set initial offset to 45 seconds and final offset to 75 seconds. Frame overlap percentage is 0.6. "
        "The end time offset is 35 minutes. Search type is CROSS_TRACK. "
        "The search is scheduled to commence 10 minutes after the objective's start time. Integration time is 1.5 seconds, and include 10 frames."
    ): {
        "binning": None,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 35,
        "final_offset": 75,
        "frame_overlap_percentage": 0.6,
        "frame_type": "LIGHT",
        "initial_offset": 45,
        "integration_time": 1.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 12, 15, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 12, 15, 8, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 1,
        "search_start_time": "datetime.datetime(2025, 12, 15, 8, 10, tzinfo=TzInfo(UTC))",  # Manual calculation: objective_start_time + 10 minutes
        "search_type": "CROSS_TRACK",
        "sensor_name": "WALLY7",
        "target_id": "alpha-007",
        "visibility_check": False,
    },
    # New Example 392: GeodssRevisitObjective
    (
        "Formulate a GeodssRevisitObjective. Target IDs are 'target-x1', 'target-y2'. Sensors are 'GEODSS-SITE1', 'GEODSS-SITE2'. "
        "The marking is U//FOUO. Data mode is EXERCISE. Priority: 15. Tracking: RATE_TRACK. "
        "Start objective: 2026-01-10 03:00:00+00:00. Readout rate: 0 (1MHz). Gain setting: 1 (Low Gain). "
        "SOI filter: 2 (10% Light). Autotrack type: 2 (Manual). Camera mode: 1 (Zoomed EBS). "
        "Array kind: 1 (Photometer). Binning mode: 0 (No Binning). Scan mode: 0 (Continuous). "
        "Number of observations is 3. Patience will be 25 minutes."
    ): {
        "acquisition_type": 0,  # Default
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "U//FOUO",
        "collect_request_type": "RATE_TRACK",
        "command": 0,  # Default
        "data_mode": "EXERCISE",
        "frame_type": "LIGHT",  # Default
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,  # Default
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 0,  # Default
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2026, 1, 10, 3, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default
        "optimal_frames_per_hour": 400,  # Default
        "overscan": 0,  # Default
        "patience_minutes": 25,
        "priority": 15,
        "rate_track_verify": 0,  # Default
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["GEODSS-SITE1", "GEODSS-SITE2"],
        "soi_filter_position": 2,
        "target_id_list": ["target-x1", "target-y2"],
        "visibility_check": False,  # Default
    },
    # New Example 393: PeriodicRevisitObjective
    (
        "Construct a PeriodicRevisitObjective. Target IDs: 'obj-101', 'obj-102', 'obj-103'. Sensors involved: 'SENSR-A', 'SENSR-B'. "
        "Classification marking is U. Data mode is TEST. Priority is 7. Patience_minutes is set to 40. "
        "Disregard other objective intent submissions. "
        "The objective's start time is 2026-02-20 12:00:00+00:00. Optimal frames per hour is 300. "
        "Specify 3 frames with an integration time of 2.5 seconds. Visibility check should be enabled. Revisits per hour: 4.5."
    ): {
        "classification_marking": "U",
        "target_id_list": ["obj-101", "obj-102", "obj-103"],
        "sensor_name_list": ["SENSR-A", "SENSR-B"],
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,
        "patience_minutes": 40,
        "revisits_per_hour": 4.5,
        "number_of_frames": 3,
        "integration_time": 2.5,
        "binning": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 12, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 300,
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 394: UctObservationObjective
    (
        "Create a UctObservationObjective. For UCT RSO IDs 'uct-delta', 'uct-gamma'. Sensors are 'OBSVR1', 'OBSVR2'. "
        "The marking is TS. Data mode: REAL. Orbital regime: XGEO. Priority: 5. Revisits per hour: 8.0. "
        "Objective starts 2026-03-05 20:00:00+00:00. Sort by brightest UCT. End time offset: 45 minutes. "
        "Visibility check is false. Number of frames: 7. Integration time: 1.0 seconds. Patience: 20 minutes."
    ): {
        "classification_marking": "TS",
        "uct_rso_id_list": ["uct-delta", "uct-gamma"],
        "sensor_name_list": ["OBSVR1", "OBSVR2"],
        "data_mode": "REAL",  # Default
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default
        "frame_type": "LIGHT",  # Default
        "orbital_regime": "XGEO",
        "visibility_check": False,
        "patience_minutes": 20,
        "revisits_per_hour": 8.0,
        "number_of_frames": 7,
        "integration_time": 1.0,
        "binning": None,
        "end_time_offset_minutes": 45,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 3, 5, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 5,
        "sort_by_brightest_uct": True,
        "objective_name": "UctObservationObjective",
    },
    # New Example 395: SingleIntentObjective
    (
        "Initiate a SingleIntentObjective. Target ID is 'unique-target-001'. RSO ID is 'rso-beta-9'. "
        "Sensors: 'SENSOR-XRAY', 'SENSOR-GAMMA'. Marking: C. Data Mode: SIMULATED. "
        "Tracking: SIDEREAL. Priority: 3. Objective begins 2026-04-10 01:15:00+00:00. "
        "Set frames to 1, integration time to 10 seconds, and binning to 4. The intent should start no earlier than 2026-04-10 01:20:00+00:00."
    ): {
        "classification_marking": "C",
        "target_id": "unique-target-001",
        "rso_id": "rso-beta-9",
        "sensor_name_list": ["SENSOR-XRAY", "SENSOR-GAMMA"],
        "data_mode": "SIMULATED",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default
        "number_of_frames": 1,
        "integration_time": 10.0,
        "priority": 3,
        "binning": 4,
        "intent_start_time": "datetime.datetime(2026, 4, 10, 1, 20, tzinfo=TzInfo(UTC))",
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 4, 10, 1, 15, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 396: DataEnrichmentObjective
    (
        "Establish a DataEnrichmentObjective. Targets: 'enrich-01', 'enrich-02'. Sensors: 'DATA-GRABBER-A'. "
        "Marking: U//FOUO. Data Mode: EXERCISE. Max RSO to observe: 10. Revisits per hour: 15.0. "
        "Objective starts 2026-05-22 06:00:00+00:00. Visibility check is false. Priority: 25. Collect request type is RATE_TRACK. "
        "Intent end time is 2026-05-22 12:00:00+00:00."
    ): {
        "classification_marking": "U//FOUO",
        "data_mode": "EXERCISE",
        "objective_uuid": None,
        "target_id_list": ["enrich-01", "enrich-02"],
        "sensor_name_list": ["DATA-GRABBER-A"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 15.0,
        "objective_start_time": "datetime.datetime(2026, 5, 22, 6, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": "datetime.datetime(2026, 5, 22, 12, 0, tzinfo=TzInfo(UTC))",
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 397: SensorCheckoutObjective
    (
        "Configure a SensorCheckoutObjective. Classification: U. Sensor: 'CHECKMATE-SENSOR'. Orbital regime: LEO. "
        "Data mode: TEST. Tracking: RATE_TRACK. Priority: 8. Revisits/hour: 0.5. "
        "Start: 2026-06-30 23:00:00+00:00. Visibility check: true. Patience: 10 minutes. "
        "Frames: 2. Integration: 5.0 seconds. Binning is 1. The intent end time is 2026-07-01 01:00:00+00:00."
    ): {
        "classification_marking": "U",
        "sensor_name": "CHECKMATE-SENSOR",
        "orbital_regime": "LEO",
        "data_mode": "TEST",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default
        "visibility_check": True,  # Default
        "patience_minutes": 10,
        "revisits_per_hour": 0.5,
        "number_of_frames": 2,
        "integration_time": 5.0,
        "binning": 1,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 6, 30, 23, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": "datetime.datetime(2026, 7, 1, 1, 0, tzinfo=TzInfo(UTC))",
        "priority": 8,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 398: BaselineAutonomyObjective
    (
        "Deploy a BaselineAutonomyObjective. UUID: 'autonomy-run-001-xyz'. Marking: S. "
        "Data mode is REAL. Frame type is DARK. Priority: 1500. "
        "Baseline autonomy RSO catalog IDs: '70001,70002,70003'. "
        "The objective has no specific end time for continuous operation. Provide RSO IDs 'rso-main-1', 'rso-main-2'."
    ): {
        "objective_uuid": "autonomy-run-001-xyz",
        "classification_marking": "S",
        "data_mode": "REAL",  # Default
        "frame_type": "DARK",
        "priority": 1500,
        "baseline_autonomy_rso": "70001,70002,70003",
        "objective_end_time": None,  # Default
        "rso_id_list": ["rso-main-1", "rso-main-2"],
        "objective_name": "BaselineAutonomyObjective",  # Default
    },
    # New Example 399: CatalogMaintenanceObjective
    (
        "Formulate a CatalogMaintenanceObjective using sensors RME10 and LMNT11. "
        "The classification marking should be S, with a data mode set to REAL. "
        "Employ the SIDEREAL tracking type within the MEO orbital regime. "
        "Assign a priority level of 500 to this objective. "
        "Configure the patience parameter to 45 minutes and the end time offset to 30 minutes. "
        "Disable the visibility check feature. "
        "The objective should commence at 2025-06-10 10:00:00+00:00 and run until 2025-06-10 14:00:00+00:00. "
        "Include the following RSO IDs in the target list: '98765' and '45678'."
    ): {
        "binning": None,
        "classification_marking": "S",
        "collect_request_type": "SIDEREAL",
        "data_mode": "REAL",
        "end_time_offset_minutes": 30,
        "frame_type": "LIGHT",
        "objective_end_time": "datetime.datetime(2025, 6, 10, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "CatalogMaintenanceObjective",
        "objective_start_time": "datetime.datetime(2025, 6, 10, 10, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "orbital_regime": "MEO",
        "patience_minutes": 45,
        "priority": 500,
        "rso_id_list": ["98765", "45678"],
        "sensor_name_list": ["RME10", "LMNT11"],
        "visibility_check": False,
    },
    # New Example 400: SearchObjective
    (
        "Generate a SearchObjective targeting RSO '77889' using sensor UKR15. "
        "Specify the classification as TS and the data mode as SIMULATED. "
        "The collect request type should be RATE_TRACK_SIDEREAL. "
        "Set the objective priority to 3. "
        "Define the initial offset before the RSO's state as 90 seconds and the final offset after as 120 seconds. "
        "Require a frame overlap percentage of 85%. "
        "The objective end time should be offset by 60 minutes from the start. "
        "The objective is scheduled to begin at 2025-07-15 20:00:00+00:00 and conclude at 2025-07-15 23:00:00+00:00. "
        "Utilize the ALONG_TRACK search type. "
        "Configure the search to start 20 minutes after the objective begins. "
        "Each intent should consist of 10 frames with an integration time of 3.5 seconds."
    ): {
        "binning": None,
        "classification_marking": "TS",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "data_mode": "SIMULATED",
        "end_time_offset_minutes": 60,
        "final_offset": 120,
        "frame_overlap_percentage": 0.85,
        "frame_type": "LIGHT",
        "initial_offset": 90,
        "integration_time": 3.5,
        "number_of_frames": 10,
        "objective_end_time": "datetime.datetime(2025, 7, 15, 23, 0, tzinfo=TzInfo(UTC))",
        "objective_name": "SearchObjective",
        "objective_start_time": "datetime.datetime(2025, 7, 15, 20, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "priority": 3,
        "search_start_time": "datetime.datetime(2025, 7, 15, 20, 20, tzinfo=TzInfo(UTC))",
        "search_type": "ALONG_TRACK",
        "sensor_name": "UKR15",
        "target_id": "77889",
        "visibility_check": False,
    },
    # New Example 401: GeodssRevisitObjective
    (
        "Initiate a GeodssRevisitObjective for targets '33445', '88990' using sensors RME20, LMNT21. "
        "Set the classification marking to C and the data mode to REAL. "
        "The collect request type should be RATE_TRACK_SIDEREAL. "
        "Assign a priority of 8 to this task. "
        "The objective is scheduled to start on 2025-08-20 21:00:00+00:00, with no specified end time for continuous operation. "
        "Configure the readout rate setting to 0 (1MHz) and the gain setting to 1 (Low Gain). "
        "Set the SOI filter position to 2 (10% Light) and the auto track type to 2 (Manual). "
        "The camera mode should be 1 (Zoomed EBS). "
        "Use array kind 1 (Photometer) and binning mode 0 (No Binning). "
        "The scan mode should be 0 (Continuous). "
        "Request 3 observations per intent."
    ): {
        "acquisition_type": 0,  # Default Auto Sidereal, not specified to change
        "array_kind": 1,
        "auto_track_roi_position": 0,  # Default Optical Boresite, not specified to change
        "auto_track_type": 2,
        "binning_mode": 0,
        "camera_mode": 1,
        "classification_marking": "C",
        "collect_request_type": "RATE_TRACK_SIDEREAL",
        "command": 0,  # Default Configure, not specified to change
        "data_mode": "REAL",
        "frame_type": "LIGHT",
        "gain_setting": 1,
        "ignore_other_objective_intent_submissions": False,
        "integration_time": None,
        "intent_end_time": None,
        "intent_start_time": None,
        "num_observations": 3,
        "num_skip_frames": 0,  # Default 0, not specified to change
        "number_of_frames": None,
        "objective_end_time": None,
        "objective_name": "GeodssRevisitObjective",
        "objective_start_time": "datetime.datetime(2025, 8, 20, 21, 0, tzinfo=TzInfo(UTC))",
        "objective_uuid": None,
        "observation_interval": 0.0,  # Default 0.0, not specified to change
        "optimal_frames_per_hour": 400,  # Default 400, not specified to change
        "overscan": 0,  # Default Normal, not specified to change
        "patience_minutes": 30,  # Default 30, not specified to change
        "priority": 8,
        "rate_track_verify": 0,  # Default No Verify, not specified to change
        "readout_rate_setting": 0,
        "revisits_per_hour": None,
        "scan_mode": 0,
        "sensor_name_list": ["RME20", "LMNT21"],
        "soi_filter_position": 2,
        "target_id_list": ["33445", "88990"],
        "visibility_check": False,  # Default False, not specified to change
    },
    # New Example 402: PeriodicRevisitObjective
    (
        "Submit a PeriodicRevisitObjective for targets '11111', '22222', '33333' using sensor UKR08. "
        "Assign a classification marking of U//FOUO and a data mode of EXERCISE. "
        "The objective's priority should be 15. "
        "Set the patience duration to 60 minutes. "
        "Ensure that submissions from other objectives are ignored. "
        "The objective is set to commence at 2025-09-05 15:30:00+00:00 and conclude at 2025-09-05 18:30:00+00:00. "
        "Specify a desired revisit rate of 3.0 revisits per hour. "
        "Each observation should capture 7 frames with an integration time of 5 seconds."
    ): {
        "classification_marking": "U//FOUO",
        "target_id_list": ["11111", "22222", "33333"],
        "sensor_name_list": ["UKR08"],
        "data_mode": "EXERCISE",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default, not specified to change
        "frame_type": "LIGHT",  # Default, not specified to change
        "visibility_check": False,  # Default, not specified to change
        "patience_minutes": 60,
        "revisits_per_hour": 3.0,
        "number_of_frames": 7,
        "integration_time": 5.0,
        "binning": None,
        "objective_start_time": "datetime.datetime(2025, 9, 5, 15, 30, tzinfo=TzInfo(UTC))",
        "objective_end_time": "datetime.datetime(2025, 9, 5, 18, 30, tzinfo=TzInfo(UTC))",
        "priority": 15,
        "ignore_other_objective_intent_submissions": True,
        "optimal_frames_per_hour": 400,  # Default, not specified to change
        "objective_uuid": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_name": "PeriodicRevisitObjective",
    },
    # New Example 403: UctObservationObjective
    (
        "Establish a UctObservationObjective for UCT RSOs '65432', '10987'. "
        "Utilize sensors RME25 and LMNT26. "
        "The classification marking is U, and the data mode is REAL. "
        "Focus on the GEO orbital regime. "
        "Set the priority level to 7. "
        "Request 4.5 revisits per hour for these UCTs. "
        "The objective should commence at 2025-10-10 09:00:00+00:00 and continue indefinitely. "
        "Set the end time offset for intents to 90 minutes. "
        "Disable the visibility check. "
        "Order the UCTs for observation without sorting by brightest. "
        "Capture 3 frames per intent with an integration time of 1.8 seconds."
    ): {
        "classification_marking": "U",
        "uct_rso_id_list": ["65432", "10987"],
        "sensor_name_list": ["RME25", "LMNT26"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK_SIDEREAL",  # Default, not specified to change
        "frame_type": "LIGHT",  # Default, not specified to change
        "orbital_regime": "GEO",
        "visibility_check": False,
        "patience_minutes": 30,  # Default, not specified to change
        "revisits_per_hour": 4.5,
        "number_of_frames": 3,
        "integration_time": 1.8,
        "binning": None,
        "end_time_offset_minutes": 90,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 10, 10, 9, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 7,
        "sort_by_brightest_uct": False,
        "objective_name": "UctObservationObjective",
    },
    # New Example 404: SingleIntentObjective
    (
        "Configure a SingleIntentObjective for target ID '99001' and RSO ID '44556'. "
        "The observation should use sensor RME30. "
        "Apply a classification marking of U//FOUO and a data mode of REAL. "
        "Use the RATE_TRACK collect request type. "
        "Assign a priority of 5. "
        "The objective should begin at 2025-11-01 14:00:00+00:00. "
        "Specify that the intent should capture 8 frames with an integration time of 4 seconds and a binning setting of 1."
    ): {
        "classification_marking": "U//FOUO",
        "target_id": "99001",
        "rso_id": "44556",
        "sensor_name_list": ["RME30"],
        "data_mode": "REAL",
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default, not specified to change
        "number_of_frames": 8,
        "integration_time": 4.0,
        "priority": 5,
        "binning": 1,
        "intent_start_time": None,
        "intent_end_time": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2025, 11, 1, 14, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "objective_name": "SingleIntentObjective",
    },
    # New Example 405: DataEnrichmentObjective
    (
        "Create a DataEnrichmentObjective for targets '10101', '20202'. "
        "Employ sensors LMNT35 and LMNT36 for data collection. "
        "The classification marking is S and the data mode is REAL. "
        "Specify the collect request type as RATE_TRACK. "
        "Set the maximum number of RSOs to observe to 10. "
        "Require 8 revisits per hour. "
        "The objective is scheduled to start at 2026-01-15 07:00:00+00:00 and run continuously. "
        "Assign a priority level of 25. "
        "Disable the visibility check functionality."
    ): {
        "classification_marking": "S",
        "data_mode": "REAL",
        "objective_uuid": None,
        "target_id_list": ["10101", "20202"],
        "sensor_name_list": ["LMNT35", "LMNT36"],
        "collect_request_type": "RATE_TRACK",
        "frame_type": "LIGHT",  # Default, not specified to change
        "binning": None,
        "max_rso_to_observe": 10,
        "revisits_per_hour": 8.0,
        "objective_start_time": "datetime.datetime(2026, 1, 15, 7, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "priority": 25,
        "intent_start_time": None,
        "intent_end_time": None,
        "visibility_check": False,
        "objective_name": "DataEnrichmentObjective",
    },
    # New Example 406: SensorCheckoutObjective
    (
        "Generate a SensorCheckoutObjective for sensor RME02. "
        "Set the classification marking to C and the data mode to REAL. "
        "The orbital regime for this checkout is LEO. "
        "The collect request type should be SIDEREAL. "
        "Assign a priority of 12. "
        "Request 2.5 revisits per hour. "
        "The objective should begin at 2026-02-20 18:00:00+00:00 and have no end time. "
        "Disable the visibility check feature. "
        "Set the patience for intents to 20 minutes. "
        "Each observation should consist of 6 frames with an integration time of 1.5 seconds."
    ): {
        "classification_marking": "C",
        "sensor_name": "RME02",
        "orbital_regime": "LEO",
        "data_mode": "REAL",
        "collect_request_type": "SIDEREAL",
        "frame_type": "LIGHT",  # Default, not specified to change
        "visibility_check": False,
        "patience_minutes": 20,
        "revisits_per_hour": 2.5,
        "number_of_frames": 6,
        "integration_time": 1.5,
        "binning": None,
        "objective_uuid": None,
        "objective_start_time": "datetime.datetime(2026, 2, 20, 18, 0, tzinfo=TzInfo(UTC))",
        "objective_end_time": None,
        "intent_start_time": None,
        "intent_end_time": None,
        "priority": 12,
        "objective_name": "SensorCheckoutObjective",
    },
    # New Example 405: BaselineAutonomyObjective, note that 405 is actual number
    (
        "Create a BaselineAutonomyObjective with UUID 'a1b2c3d4-e5f6-7890-1234-567890abcdef'. "
        "Use S markings, TEST mode, and DARK frame type. "
        "Set the priority to 50. "
        "Include RSO IDs '44444', '55555' and catalog IDs '21220', '23240', '25260'. "
        "The objective should conclude at 2026-03-10 12:00:00+00:00."
    ): {
        "objective_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
        "classification_marking": "S",
        "data_mode": "TEST",
        "frame_type": "DARK",
        "priority": 50,
        "baseline_autonomy_rso": "21220,23240,25260",
        "objective_end_time": "datetime.datetime(2026, 3, 10, 12, 0, tzinfo=TzInfo(UTC))",
        "rso_id_list": ["44444", "55555"],
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


categorized_errors_template = {
    "Missing Field Data": {
        "description": 'The system predicted the literal string "MISSING" for a field where a value was expected.',
        "examples": [
            'e.g., {"field": "objective_name", "expected": "SomeName", "predicted": "MISSING"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Null When Value Expected": {
        "description": "The system predicted `null` (absence of value) when a specific, non-null value was expected.",
        "examples": [
            'e.g., {"field": "objective_start_time", "expected": "datetime(...)", "predicted": null}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Value Present When Null Expected": {
        "description": "The system predicted a concrete value for a field that was expected to be `null`.",
        "examples": [
            'e.g., {"field": "intent_start_time", "expected": null, "predicted": "2025-09-05T10:00:00Z"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Datetime Formatting Mismatch (Equivalent Value)": {
        "description": "Expected (e.g., Python `datetime` string) and predicted (e.g., ISO 8601 string) represent the exact same point in time but differ in format.",
        "examples": [
            'e.g., {"field": "intent_start_time", "expected": "datetime(2026,2,20,14,30,...)", "predicted": "2026-02-20T14:30:00+00:00"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Datetime Value Mismatch": {
        "description": "Both expected and predicted are datetime representations but refer to different points in time (different hour, day, etc.).",
        "examples": [
            'e.g., {"field": "search_start_time", "expected": "datetime(...,16,50,...)", "predicted": "...T18:50:00Z"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Incorrect Semantic String - Abbreviation/Full Name": {
        "description": "A string mismatch where one value is an abbreviation and the other is its full form (e.g., classification markings).",
        "examples": [
            'e.g., {"field": "classification_marking", "expected": "C", "predicted": "CONFIDENTIAL"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Incorrect Semantic String - Related/Distinct Terms": {
        "description": "A string mismatch where terms are related but distinct, indicating a mix-up in allowed values or vocabulary (e.g., types of data collection).",
        "examples": [
            'e.g., {"field": "collect_request_type", "expected": "SIDEREAL", "predicted": "RATE_TRACK_SIDEREAL"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Incorrect Semantic String - Arbitrary Mismatch": {
        "description": "A string mismatch where the predicted string is substantially different from the expected string with no clear semantic link.",
        "examples": [
            'e.g., {"field": "sensor_name", "expected": "ZAP55", "predicted": "AWESOME_SENSOR"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Numerical Value Mismatch": {
        "description": "Both expected and predicted are numerical types (int, float), but their values differ.",
        "examples": [
            'e.g., {"field": "number_of_frames", "expected": 8, "predicted": 40}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Case Sensitivity Mismatch (String)": {
        "description": "Expected and predicted string values are identical if letter casing is ignored, but differ in actual casing.",
        "examples": [
            'e.g., {"field": "sensor_name", "expected": "vista-b", "predicted": "Vista-B"}'
        ],
        "total_count": 0,
        "field_counts": {},
    },
    "Boolean Value Mismatch": {
        "description": "Expected and predicted values represent different boolean states (true vs. false).",
        "examples": [
            'e.g., {"field": "visibility_check", "expected": true, "predicted": false}'
        ],  # Assuming actual booleans
        "total_count": 0,
        "field_counts": {},
    },
    "Probable Field Value Swapping/Misplacement (Datetime)": {
        "description": (
            "This error on an 'intent_*_time' field (value present when null expected) is likely part of a pair. "
            "The corresponding 'objective_*_time' field was likely null (but expected a value), and this 'intent' field wrongly received that objective's value."
        ),
        "examples": [
            "e.g., 'intent_start_time' predicted '2025-09-05T10:00Z' (expected null), while related 'objective_start_time' was null (expected 'datetime(2025,9,5,10,0,...Z)')"
        ],
        "total_count": 0,
        "field_counts": {},  # Tracks the 'intent_*_time' fields involved
    },
    "OTHER": {
        "description": "Errors that do not fit into any of the other predefined categories.",
        "examples": [],
        "total_count": 0,
        "field_counts": {},
    },
}


# --- Helper Functions ---
def is_python_datetime_repr(value):
    """Checks if a value is a string representation of a Python datetime object."""
    return isinstance(value, str) and value.startswith("datetime.datetime(")


def is_iso_datetime_str(value):
    """Checks if a value is an ISO 8601 formatted datetime string."""
    if not isinstance(value, str):
        return False
    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
    return bool(re.match(iso_pattern, value))


def are_datetimes_equivalent(expected_dt_repr, predicted_dt_iso_str):
    """Simplified check for datetime equivalence.
    WARNING: This is a placeholder. Robust parsing (e.g., with dateutil.parser)
    and timezone-aware comparison are needed for a production system.
    """
    if not is_python_datetime_repr(expected_dt_repr) or not is_iso_datetime_str(
        predicted_dt_iso_str
    ):
        return False
    try:
        # Extremely simplified extraction - only compares main components if they match a simple pattern
        match_expected = re.search(
            r"(\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})",
            expected_dt_repr,
        )
        # Simplification: Assumes predicted is already UTC if it has Z or +00:00, or no offset means to compare naively
        match_predicted = re.search(
            r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})", predicted_dt_iso_str
        )

        if match_expected and match_predicted:
            # Compare year, month, day, hour, minute, second naively
            return all(
                int(match_expected.group(i)) == int(match_predicted.group(i))
                for i in range(1, 7)
            )
    except ValueError:  # Handles int() conversion errors
        return False
    return False


def get_comparable_datetime_value_from_repr(dt_repr_str):
    """Simplified: Extracts YYYY-MM-DDTHH:MM:SS from Python datetime repr for basic comparison."""
    if not is_python_datetime_repr(dt_repr_str):
        return None
    match = re.search(
        r"(\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})",
        dt_repr_str,
    )
    if match:
        parts = [int(p) for p in match.groups()]
        return f"{parts[0]:04}-{parts[1]:02}-{parts[2]:02}T{parts[3]:02}:{parts[4]:02}:{parts[5]:02}"
    return None


def get_comparable_datetime_value_from_iso(dt_iso_str):
    """Simplified: Extracts YYYY-MM-DDTHH:MM:SS from ISO str for basic comparison."""
    if not is_iso_datetime_str(dt_iso_str):
        return None
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", dt_iso_str)
    if match:
        return match.group(1)
    return None


def is_number(value):
    """Checks if a value is an integer or float."""
    return isinstance(value, (int, float))


# --- Classification Function ---
def classify_error(field_name, expected, predicted, current_run_field_details=None):
    """Classifies a single error into one of the defined categories.
    `current_run_field_details` (dict of all fields for the current run/record)
    is needed for Category 12 (Swapping/Misplacement).
    """
    if current_run_field_details is None:
        current_run_field_details = {}

    # Category 1: Missing Field Data
    if predicted == "MISSING":
        return "Missing Field Data"

    # Category 12: Probable Field Value Swapping/Misplacement (Datetime)
    # This specifically identifies the 'intent_*_time' part of a potential swap.
    if (
        field_name in ["intent_start_time", "intent_end_time"]
        and expected is None
        and predicted is not None
        and is_iso_datetime_str(predicted)
    ):
        objective_field_equivalent = field_name.replace("intent_", "objective_")
        if objective_field_equivalent in current_run_field_details:
            obj_info = current_run_field_details[objective_field_equivalent]
            # Check if objective field was 'Null When Value Expected' and its expected value matches this intent's predicted value
            if (
                obj_info.get("predicted") is None
                and obj_info.get("expected") is not None
                and is_python_datetime_repr(obj_info["expected"])
            ):
                intent_pred_comparable = get_comparable_datetime_value_from_iso(
                    predicted
                )
                obj_expected_comparable = get_comparable_datetime_value_from_repr(
                    obj_info["expected"]
                )

                if (
                    intent_pred_comparable
                    and obj_expected_comparable
                    and intent_pred_comparable == obj_expected_comparable
                ):
                    return "Probable Field Value Swapping/Misplacement (Datetime)"

    # Category 2: Null When Value Expected
    if (
        predicted is None and expected is not None
    ):  # 'expected' can be 0 or False, which are not None
        return "Null When Value Expected"

    # Category 3: Value Present When Null Expected (and not caught by Cat 12)
    if predicted is not None and expected is None:
        return "Value Present When Null Expected"

    # Datetime Categories (4 and 5)
    is_expected_dt_python_repr = is_python_datetime_repr(expected)
    is_predicted_dt_iso_str = is_iso_datetime_str(predicted)

    if is_expected_dt_python_repr and is_predicted_dt_iso_str:
        if are_datetimes_equivalent(expected, predicted):
            return "Datetime Formatting Mismatch (Equivalent Value)"
        else:
            return "Datetime Value Mismatch"
    # Consider if expected is ISO and predicted is Python repr (less common in sample)
    # For now, assuming one primary direction of datetime format difference.

    # String categories
    if isinstance(expected, str) and isinstance(predicted, str):
        # Category 8: Case Sensitivity Mismatch (must be checked before other string categories)
        if expected.lower() == predicted.lower() and expected != predicted:
            return "Case Sensitivity Mismatch (String)"

        # Category 6A: Abbreviation/Full Name
        if field_name == "classification_marking":
            abbrev_pairs = {
                ("C", "CONFIDENTIAL"),
                ("S", "SECRET"),
                ("TS", "TOP SECRET"),
            }
            # Check if (expected, predicted) or (predicted, expected) is in the set to cover both directions
            if any(
                (e, p) in abbrev_pairs or (p, e) in abbrev_pairs
                for e_abbr, p_full in abbrev_pairs
                for e, p in [(expected, predicted)]
            ):
                return "Incorrect Semantic String - Abbreviation/Full Name"

        # Category 6B: Related/Distinct Terms
        if field_name == "collect_request_type":
            crt_pairs = {
                ("SIDEREAL", "RATE_TRACK_SIDEREAL"),
                ("RATE_TRACK", "RATE_TRACK_SIDEREAL"),
                ("RATE_TRACK_SIDEREAL", "RATE_TRACK"),
            }
            if (expected, predicted) in crt_pairs:
                return "Incorrect Semantic String - Related/Distinct Terms"
        if (
            field_name == "frame_type"
            and expected == "LIGHT"
            and predicted in ["STANDARD", "DEFAULT"]
        ):
            return "Incorrect Semantic String - Related/Distinct Terms"
        if (
            field_name == "search_type"
            and expected == "RASTER_SCAN"
            and predicted == "SIDEREAL"
        ):
            return "Incorrect Semantic String - Related/Distinct Terms"

        # Category 6C: Arbitrary String Mismatch (if none of the above string issues or datetime issues apply)
        # This is a fallback for general string mismatches not fitting specific patterns.
        # Ensure not to re-classify already identified datetime format issues as arbitrary string issues.
        if not (
            is_expected_dt_python_repr or is_predicted_dt_iso_str
        ):  # Avoids re-classifying datetime strings
            return "Incorrect Semantic String - Arbitrary Mismatch"

    # Category 7: Numerical Value Mismatch
    if is_number(expected) and is_number(predicted) and expected != predicted:
        return "Numerical Value Mismatch"

    # Category 9: Boolean Value Mismatch
    # Handles actual booleans. If "true"/"false" strings, conversion would be needed.
    if (
        isinstance(expected, bool)
        and isinstance(predicted, bool)
        and expected != predicted
    ):
        return "Boolean Value Mismatch"

    return "OTHER"


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
        # all_errors = []
        # for r in self.results:
        #     for field_name, info in r["field_details"].items():
        #         if not info["correct"]:
        #             all_errors.append(
        #                 {
        #                     "field": field_name,
        #                     "expected": info["expected"],
        #                     "predicted": info["predicted"],
        #                 }
        #             )

        categorized_errors_summary = copy.deepcopy(categorized_errors_template)

        for r in self.results:
            current_run_field_details = r.get("field_details", {})

            for field_name, info in current_run_field_details.items():
                # Process only if the 'correct' flag is explicitly False
                if info.get("correct") is False:
                    expected_val = info.get("expected")
                    predicted_val = info.get("predicted")

                    category_name = classify_error(
                        field_name,
                        expected_val,
                        predicted_val,
                        current_run_field_details,
                    )

                    # Ensure category_name is valid, defaulting to "OTHER" if somehow not found
                    if category_name not in categorized_errors_summary:
                        print(
                            f"Warning: Unknown category '{category_name}' for field '{field_name}'. Defaulting to OTHER."
                        )
                        category_name = "OTHER"

                    # Update counts for the determined category
                    categorized_errors_summary[category_name]["total_count"] += 1
                    categorized_errors_summary[category_name]["field_counts"][
                        field_name
                    ] = (
                        categorized_errors_summary[category_name]["field_counts"].get(
                            field_name, 0
                        )
                        + 1
                    )

        report["categorized_error_summary"] = categorized_errors_summary
        # report["all_errors"] = all_errors

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
