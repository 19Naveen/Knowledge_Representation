import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryExecuteRequest(BaseModel):
    dataset_id: uuid.UUID
    sql: str = Field(min_length=1)
    row_limit: int = Field(default=1000, ge=1, le=10000)


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class AggregateRequest(BaseModel):
    dataset_id: uuid.UUID
    dimension: str
    measure: str | None = None
    aggregation: Literal["sum", "avg", "count", "min", "max"] = "sum"
    limit: int = Field(default=100, ge=1, le=1000)


class AggregatePoint(BaseModel):
    label: str | None
    value: float | int | None


class AggregateResponse(BaseModel):
    data: list[AggregatePoint]


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    dataset_schema: dict[str, str]
