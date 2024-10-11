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
