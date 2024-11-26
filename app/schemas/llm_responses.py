from enum import Enum
from typing import List, Optional

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


# version 1
# class ChunkMetadata(BaseModel):
#     source_files: List[str] = Field(
#         ..., description="List of source JSON files contributing to this chunk"
#     )
#     json_keys_summary: List[str] = Field(
#         ...,
#         description="List of key names appearing in the chunk, providing a quick summary of the contents",
#     )
#     descriptive_labels: Dict[str, str] = Field(
#         ...,
#         description="Mapping of JSON keys to more understandable labels within the chunk, if applicable",
#     )
#     context_info: str = Field(
#         None,
#         description="Additional context or notes that describe the overall content of this chunk",
#     )
#     num_values: int = Field(
#         ..., description="Number of JSON key-value pairs in the chunk"
#     )
#     priority_level: int = Field(
#         1,
#         ge=1,
#         le=100,
#         description="Indicates the priority level of the chunk. 1 is low priority and 100 is highest priority",
#     )


class Category(str, Enum):
    OBSERVATION_REQUEST = "ObservationRequest"
    SPACE_OBJECT = "SpaceObject"
    SENSOR_INFO = "SensorInfo"
    INSTRUMENT_INFO = "InstrumentInfo"
    OBSERVATION_CONSTRAINT = "ObservationConstraint"
    STATUS_UPDATE = "StatusUpdate"
    OPERATIONAL_STATUS = "OperationalStatus"
    SCHEDULING_INFO = "SchedulingInfo"


# version 2
class ChunkMetadata(BaseModel):
    source_files: List[str] = Field(
        ..., description="List of source files contributing to this chunk."
    )
    categories: List[Category] = Field(
        ...,
        description="List of high-level categories or tags applicable to this chunk.",
    )
    summary: str = Field(
        ...,
        description="A concise summary (2-4 sentences) explaining the chunk's content.",
    )
    key_points: Optional[List[str]] = Field(
        None, description="List of key points or highlights from the chunk."
    )
    context_info: Optional[str] = Field(
        None, description="Additional context or notes about the chunk."
    )
    priority_level: int = Field(
        1,
        ge=1,
        le=100,
        description="Priority level of the chunk (1=lowest priority, 100=highest priority).",
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
