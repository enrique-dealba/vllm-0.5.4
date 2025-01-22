from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BasicLLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")


class DetailedLLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")
    sources: Optional[List[str]] = Field(
        None, description="Sources or references for the response"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the response"
    )


# TODO: Test "How confident you are about your response"
class EvidenceLLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")
    evidence: List[str] = Field(
        ...,
        description="Verbatim JSON key-value pairs supporting the response. Format: 'key: value'. Use dot notation for nested structures, e.g., 'status_counts.FAILED: 3'.",
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the response"
    )


class ObjectiveType(BaseModel):
    objective_name: str = Field(
        ...,
        description="Type of objective being specified. Choose from: CatalogMaintenanceObjective, PeriodicRevisitObjective, SearchObjective, DataEnrichmentObjective, SpectralClearingObjective",
    )


class CatalogMaintenanceObjective(BaseModel):
    classification_marking: str = Field(
        ...,
        description="Classification level of objective intents. Choose from: U, C, S, TS, U//FOUO",
    )
    data_mode: str = Field(
        ...,
        description="String type for the Machina Common DataModeType. Choose from: TEST, REAL, SIMULATED, EXERCISE",
    )
    collect_request_type: str = Field(
        "RATE_TRACK_SIDEREAL",
        description="Collect request type of tracking type. Choose from: RATE_TRACK, SIDEREAL, RATE_TRACK_SIDEREAL",
    )
    orbital_regime: str = Field(
        ...,
        description="Orbital regime classification for this catalog maintenance objective. Choose from: LEO, MEO, GEO, XGEO",
    )
    patience_minutes: int = Field(
        30,
        description="Amount of time in minutes to wait before assuming an intent has failed",
    )
    end_time_offset_minutes: int = Field(
        20,
        description="Number of minutes into the future to schedule this intent",
    )
    priority: int = Field(
        1000,
        description="Priority level for scheduling (higher numbers indicate lower priority, defaults to 1000)",
    )


class ChunkMetadata(BaseModel):
    source_files: List[str] = Field(
        ..., description="List of source JSON files contributing to this chunk"
    )
    json_keys_summary: List[str] = Field(
        ...,
        description="List of key names appearing in the chunk, providing a quick summary of the contents",
    )
    descriptive_labels: Dict[str, str] = Field(
        ...,
        description="Mapping of JSON keys to more understandable labels within the chunk, if applicable",
    )
    context_info: str = Field(
        None,
        description="Additional context or notes that describe the overall content of this chunk",
    )
    num_values: int = Field(
        ..., description="Number of JSON key-value pairs in the chunk"
    )
    priority_level: int = Field(
        1,
        ge=1,
        le=100,
        description="Indicates the priority level of the chunk. 1 is low priority and 100 is highest priority",
    )


class IntentStatusSummary(BaseModel):
    target_name: str = Field(..., description="Name of the target satellite")
    target_catalog_id: str = Field(
        ..., description="Catalog ID of the target satellite"
    )

    failed_count: int = Field(
        ..., ge=0, description="Number of intents with FAILED status"
    )
    scheduled_count: int = Field(
        ..., ge=0, description="Number of intents with SCHEDULED status"
    )
    completed_count: int = Field(
        ..., ge=0, description="Number of intents with COMPLETED status"
    )

    failure_reason: Optional[str] = Field(
        None, description="Common reason for failed intents"
    )

    priority: int = Field(..., ge=0, description="Priority level of the intents")

    frame_type: str = Field(..., description="Type of frame used in the observation")
    num_frames: int = Field(
        ..., ge=0, description="Number of frames in the observation"
    )
    integration_time_s: float = Field(
        ..., ge=0, description="Integration time in seconds"
    )
    track_type: str = Field(
        ..., description="Type of tracking used for the observation"
    )

    scheduling_process: str = Field(
        ..., description="Description of the typical scheduling process"
    )
    completion_note: Optional[str] = Field(
        None, description="Note on the completed intent, if any"
    )


class IntentAnalysisSummary(BaseModel):
    summary: IntentStatusSummary = Field(
        ..., description="Detailed summary of intent statuses"
    )
    total_intents: int = Field(
        ..., ge=0, description="Total number of intents analyzed"
    )
    analysis_date: str = Field(..., description="Date of the intent analysis")
