from typing import Dict, List, Optional, Union

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

class Measurement(BaseModel):
    value: float = Field(..., description="The numerical value of the measurement")
    unit: str = Field(..., description="The unit of measurement")

class Fact(BaseModel):
    content: str = Field(..., description="The content of the fact")
    category: Optional[str] = Field(None, description="Category of the fact")

class Statistic(BaseModel):
    name: str = Field(..., description="Name of the statistic")
    value: Union[int, float, str] = Field(..., description="Value of the statistic")
    unit: Optional[str] = Field(None, description="Unit of the statistic, if applicable")

class Component(BaseModel):
    name: str = Field(..., description="Name of the component")
    description: str = Field(..., description="Description of the component")
    specifications: Optional[Dict[str, Union[str, int, float]]] = Field(None, description="Specifications of the component")

class InformationalContent(BaseModel):
    title: str = Field(..., description="Title of the informational content")
    summary: str = Field(..., description="Brief summary or overview")
    key_facts: List[Fact] = Field(..., description="List of key facts")
    statistics: List[Statistic] = Field(..., description="List of important statistics")
    components: Optional[List[Component]] = Field(None, description="List of major components or systems")
    measurements: Optional[Dict[str, Measurement]] = Field(None, description="Key measurements")
    historical_events: Optional[List[str]] = Field(None, description="List of significant historical events")
    comparisons: Optional[List[str]] = Field(None, description="Interesting comparisons or analogies")
    images: Optional[List[str]] = Field(None, description="Descriptions of related images")
    additional_info: Optional[List[str]] = Field(None, description="Any additional interesting information")
