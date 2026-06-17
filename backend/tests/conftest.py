from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from modules.ingestion.enums import TransformType


def make_version(version, schema, **kw):
    """Build a fake DatasetVersion ORM row as a SimpleNamespace (no DB needed)."""
    defaults = {
        "id": kw.get("id", 1),
        "dataset_id": kw.get("dataset_id", 1),
        "version": version,
        "storage_path": kw.get("storage_path", f"datasets/1/v{version}.parquet"),
        "row_count": kw.get("row_count", 0),
        "column_count": kw.get("column_count", len(schema)),
        "schema": schema,
        "file_size": kw.get("file_size", 0),
        "created_at": kw.get("created_at", datetime.now(timezone.utc)),
    }
    return SimpleNamespace(**defaults)


def make_rule(transform_type, source_column, target_column=None, cast_to_type=None):
    """Build a fake MappingRule ORM row as a SimpleNamespace (no DB needed)."""
    if not isinstance(transform_type, TransformType):
        transform_type = TransformType(transform_type)
    return SimpleNamespace(
        transform_type=transform_type,
        source_column=source_column,
        target_column=target_column,
        cast_to_type=cast_to_type,
    )


@pytest.fixture
def version_factory():
    return make_version


@pytest.fixture
def rule_factory():
    return make_rule
