import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


ColumnType = Literal["string", "integer", "decimal", "boolean", "timestamp"]


class DatasetResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str | None
    source_type: str
    created_at: datetime
    version_count: int = 0
    latest_row_count: int | None = None
    latest_file_size: int | None = None

    model_config = {"from_attributes": True}


class PostgresSourceConfig(BaseModel):
    host: str
    port: int = 5432
    database: str
    user: str
    password: str
    table: str
    query: str | None = None


class CreateIngestionJobRequest(BaseModel):
    dataset_id: uuid.UUID
    dataset_name: str = Field(min_length=1, max_length=255)
    workspace_id: uuid.UUID
    source_type: Literal["csv", "xlsx", "parquet", "postgres", "snowflake", "mysql", "mssql"]
    db_config: PostgresSourceConfig | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_db_config(self) -> "CreateIngestionJobRequest":
        db_sources = {"postgres", "snowflake", "mysql", "mssql"}
        if self.source_type in db_sources and self.db_config is None:
            raise ValueError("db_config is required for database sources")
        return self


class IngestionJobResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    source_type: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ColumnDiff(BaseModel):
    column: str
    old_type: str | None = None
    new_type: str | None = None
    suggested_target: str | None = None


class SchemaDiffResponse(BaseModel):
    dataset_id: uuid.UUID
    has_diff: bool
    added_columns: list[str]
    missing_columns: list[str]
    type_changes: list[ColumnDiff]
    suggested_mappings: list[ColumnDiff]


class MappingRuleRequest(BaseModel):
    source_column: str
    target_column: str | None = None
    transform_type: Literal["map", "drop", "cast"]
    cast_to_type: ColumnType | None = None

    @model_validator(mode="after")
    def validate_rule(self) -> "MappingRuleRequest":
        if self.transform_type == "cast" and self.cast_to_type is None:
            raise ValueError("cast_to_type is required when transform_type is 'cast'")
        if self.transform_type == "drop" and self.target_column is not None:
            raise ValueError("target_column must be None when transform_type is 'drop'")
        return self


class ResolveSchemaMappingRequest(BaseModel):
    rules: list[MappingRuleRequest]


class MappingRuleResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    source_column: str
    target_column: str | None = None
    transform_type: str
    cast_to_type: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetVersionResponse(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    id: uuid.UUID
    dataset_id: uuid.UUID
    version: int
    storage_path: str
    row_count: int
    column_count: int
    dataset_schema: dict[str, str] = Field(alias="schema")
    file_size: int
    created_at: datetime
