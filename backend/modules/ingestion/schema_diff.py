from dataclasses import dataclass, field

import pandas as pd

from modules.ingestion.models import MappingRule
from modules.ingestion.enums import TransformType

CAST_TYPE_MAP: dict[str, str] = {
    "string": "object",
    "integer": "int64",
    "decimal": "float64",
    "boolean": "bool",
    "timestamp": "datetime64[ns]",
}


@dataclass
class SchemaDiff:
    has_diff: bool
    added_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    type_changes: list[dict] = field(default_factory=list)
    suggested_mappings: list[dict] = field(default_factory=list)


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_diff(old_schema: dict[str, str], new_schema: dict[str, str]) -> SchemaDiff:
    old_cols = set(old_schema)
    new_cols = set(new_schema)

    added = sorted(new_cols - old_cols)
    missing = sorted(old_cols - new_cols)
    type_changes = [
        {"column": col, "old_type": old_schema[col], "new_type": new_schema[col]}
        for col in old_cols & new_cols
        if old_schema[col] != new_schema[col]
    ]

    # Suggest mappings: for each missing col, find the closest added col by name
    suggested = []
    for m_col in missing:
        best, best_dist = None, 4  # threshold: edit distance < 4
        for a_col in added:
            d = _edit_distance(m_col.lower(), a_col.lower())
            if d < best_dist:
                best, best_dist = a_col, d
        if best:
            suggested.append({"column": m_col, "suggested_target": best})

    has_diff = bool(added or missing or type_changes)
    return SchemaDiff(
        has_diff=has_diff,
        added_columns=added,
        missing_columns=missing,
        type_changes=type_changes,
        suggested_mappings=suggested,
    )


def apply_rules(df: pd.DataFrame, rules: list[MappingRule]) -> pd.DataFrame:
    df = df.copy()
    for rule in rules:
        if rule.transform_type == TransformType.DROP:
            if rule.source_column in df.columns:
                df = df.drop(columns=[rule.source_column])

        elif rule.transform_type == TransformType.MAP:
            if rule.source_column in df.columns and rule.target_column:
                df = df.rename(columns={rule.source_column: rule.target_column})

        elif rule.transform_type == TransformType.CAST:
            col = rule.source_column
            if col in df.columns and rule.cast_to_type:
                target_dtype = CAST_TYPE_MAP.get(rule.cast_to_type)
                if target_dtype:
                    if rule.cast_to_type == "timestamp":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(target_dtype, errors="ignore")

    return df
