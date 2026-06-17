import pandas as pd
import pytest

from modules.ingestion.enums import TransformType
from modules.ingestion.schema_diff import apply_rules, compute_diff

from tests.conftest import make_rule


# --------------------------------------------------------------------------- #
# compute_diff
# --------------------------------------------------------------------------- #
def test_identical_schemas_have_no_diff():
    schema = {"id": "integer", "name": "string"}
    diff = compute_diff(schema, dict(schema))

    assert diff.has_diff is False
    assert diff.added_columns == []
    assert diff.missing_columns == []
    assert diff.type_changes == []
    assert diff.suggested_mappings == []


def test_added_column_is_detected_and_sorted():
    old = {"id": "integer"}
    new = {"id": "integer", "zeta": "string", "alpha": "string"}
    diff = compute_diff(old, new)

    assert diff.has_diff is True
    assert diff.added_columns == ["alpha", "zeta"]  # sorted
    assert diff.missing_columns == []


def test_missing_column_is_detected():
    old = {"id": "integer", "email": "string"}
    new = {"id": "integer"}
    diff = compute_diff(old, new)

    assert diff.has_diff is True
    assert diff.missing_columns == ["email"]
    assert diff.added_columns == []


def test_type_change_is_detected():
    old = {"id": "integer", "amount": "integer"}
    new = {"id": "integer", "amount": "decimal"}
    diff = compute_diff(old, new)

    assert diff.has_diff is True
    assert diff.type_changes == [
        {"column": "amount", "old_type": "integer", "new_type": "decimal"}
    ]
    assert diff.added_columns == []
    assert diff.missing_columns == []


def test_suggested_mapping_for_close_name():
    # missing `amount` + added `amounts` (edit distance 1) -> suggested
    old = {"amount": "integer"}
    new = {"amounts": "integer"}
    diff = compute_diff(old, new)

    assert diff.missing_columns == ["amount"]
    assert diff.added_columns == ["amounts"]
    assert diff.suggested_mappings == [
        {"column": "amount", "suggested_target": "amounts"}
    ]


def test_no_suggestion_for_far_away_name():
    # missing `date`, only far-away added column -> no suggestion
    old = {"date": "timestamp"}
    new = {"customer_identifier": "string"}
    diff = compute_diff(old, new)

    assert diff.missing_columns == ["date"]
    assert diff.added_columns == ["customer_identifier"]
    assert diff.suggested_mappings == []


def test_suggested_mapping_is_case_insensitive():
    old = {"Amount": "integer"}
    new = {"amounts": "integer"}
    diff = compute_diff(old, new)

    assert diff.suggested_mappings == [
        {"column": "Amount", "suggested_target": "amounts"}
    ]


# --------------------------------------------------------------------------- #
# apply_rules
# --------------------------------------------------------------------------- #
@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "drop_me": [10, 20, 30],
            "old_name": ["a", "b", "c"],
            "when": ["2020-01-01", "2021-06-15", "2022-12-31"],
        }
    )


def test_drop_rule_removes_column(df):
    rules = [make_rule(TransformType.DROP, "drop_me")]
    out = apply_rules(df, rules)

    assert "drop_me" not in out.columns
    assert "id" in out.columns


def test_map_rule_renames_column(df):
    rules = [make_rule(TransformType.MAP, "old_name", target_column="new_name")]
    out = apply_rules(df, rules)

    assert "old_name" not in out.columns
    assert "new_name" in out.columns
    assert list(out["new_name"]) == ["a", "b", "c"]


def test_cast_rule_to_integer(df):
    rules = [make_rule(TransformType.CAST, "id", cast_to_type="integer")]
    out = apply_rules(df, rules)

    assert out["id"].dtype == "int64"
    assert list(out["id"]) == [1, 2, 3]


def test_cast_rule_to_timestamp(df):
    rules = [make_rule(TransformType.CAST, "when", cast_to_type="timestamp")]
    out = apply_rules(df, rules)

    assert pd.api.types.is_datetime64_any_dtype(out["when"])
    assert out["when"].iloc[0] == pd.Timestamp("2020-01-01")


def test_drop_rule_missing_column_is_noop(df):
    rules = [make_rule(TransformType.DROP, "does_not_exist")]
    out = apply_rules(df, rules)

    assert list(out.columns) == list(df.columns)


def test_map_rule_missing_column_is_noop(df):
    rules = [make_rule(TransformType.MAP, "ghost", target_column="phantom")]
    out = apply_rules(df, rules)

    assert list(out.columns) == list(df.columns)
    assert "phantom" not in out.columns


def test_apply_rules_does_not_mutate_input(df):
    original_cols = list(df.columns)
    apply_rules(df, [make_rule(TransformType.DROP, "drop_me")])

    assert list(df.columns) == original_cols
