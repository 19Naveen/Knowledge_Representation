"""Transform engine for the import wizard.

A transform plan is an ordered list of typed steps (drop / rename / cast / filter / fillna)
that the user builds in DataForge and the ingestion pipeline applies to the incoming DataFrame
before a version is written. Pure: DataFrame in → DataFrame out, no DB or storage.

Public API:
    parse_steps(raw: list[dict]) -> list[TransformStep]   # validate a plan
    apply_transforms(df, steps) -> pd.DataFrame           # steps may be dicts or parsed models
"""

from typing import Annotated, Any, Literal, Union

import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter, model_validator

# Canonical type → pandas dtype (mirrors schema_diff.CAST_TYPE_MAP).
CAST_TYPE_MAP: dict[str, str] = {
    "string": "object",
    "integer": "int64",
    "decimal": "float64",
    "boolean": "bool",
    "timestamp": "datetime64[ns]",
}


class DropStep(BaseModel):
    type: Literal["drop"]
    column: str


class RenameStep(BaseModel):
    type: Literal["rename"]
    column: str
    to: str


class CastStep(BaseModel):
    type: Literal["cast"]
    column: str
    to_type: Literal["string", "integer", "decimal", "boolean", "timestamp"]


class FilterStep(BaseModel):
    type: Literal["filter"]
    column: str
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "isnull", "notnull"]
    value: Any = None

    @model_validator(mode="after")
    def _require_value(self) -> "FilterStep":
        if self.op not in ("isnull", "notnull") and self.value is None:
            raise ValueError(f"filter op '{self.op}' requires a value")
        return self


class FillnaStep(BaseModel):
    type: Literal["fillna"]
    column: str
    strategy: Literal["value", "mean", "median", "mode"]
    value: Any = None

    @model_validator(mode="after")
    def _require_value(self) -> "FillnaStep":
        if self.strategy == "value" and self.value is None:
            raise ValueError("fillna strategy 'value' requires a value")
        return self


TransformStep = Annotated[
    Union[DropStep, RenameStep, CastStep, FilterStep, FillnaStep],
    Field(discriminator="type"),
]

_STEP_ADAPTER: TypeAdapter = TypeAdapter(TransformStep)
_PLAN_ADAPTER: TypeAdapter = TypeAdapter(list[TransformStep])

_STEP_MODELS = (DropStep, RenameStep, CastStep, FilterStep, FillnaStep)


def parse_steps(raw: list[dict]) -> list:
    """Validate a raw transform plan into typed step models (raises ValidationError)."""
    return _PLAN_ADAPTER.validate_python(raw)


def _coerce(step) -> Any:
    return step if isinstance(step, _STEP_MODELS) else _STEP_ADAPTER.validate_python(step)


_FILTER_OPS = {
    "eq": lambda s, v: s == v,
    "ne": lambda s, v: s != v,
    "lt": lambda s, v: s < v,
    "le": lambda s, v: s <= v,
    "gt": lambda s, v: s > v,
    "ge": lambda s, v: s >= v,
    "isnull": lambda s, v: s.isna(),
    "notnull": lambda s, v: s.notna(),
}


def apply_transforms(df: pd.DataFrame, steps) -> pd.DataFrame:
    """Apply transform steps in order, returning a new DataFrame (input untouched)."""
    out = df.copy()
    for raw_step in steps:
        step = _coerce(raw_step)

        if isinstance(step, DropStep):
            if step.column in out.columns:
                out = out.drop(columns=[step.column])

        elif isinstance(step, RenameStep):
            if step.column in out.columns:
                out = out.rename(columns={step.column: step.to})

        elif isinstance(step, CastStep):
            col = step.column
            if col in out.columns:
                if step.to_type == "timestamp":
                    out[col] = pd.to_datetime(out[col], errors="coerce")
                else:
                    target = CAST_TYPE_MAP[step.to_type]
                    try:
                        out[col] = out[col].astype(target, errors="ignore")
                    except TypeError:
                        try:
                            out[col] = out[col].astype(target)
                        except (ValueError, TypeError):
                            pass

        elif isinstance(step, FilterStep):
            if step.column in out.columns:
                mask = _FILTER_OPS[step.op](out[step.column], step.value)
                out = out[mask].copy()

        elif isinstance(step, FillnaStep):
            col = step.column
            if col in out.columns:
                if step.strategy == "value":
                    fill = step.value
                elif step.strategy == "mean":
                    fill = out[col].mean()
                elif step.strategy == "median":
                    fill = out[col].median()
                else:  # mode
                    modes = out[col].mode()
                    fill = modes.iloc[0] if not modes.empty else None
                if fill is not None:
                    out[col] = out[col].fillna(fill)

    return out
