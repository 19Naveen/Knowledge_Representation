import pandas as pd

DTYPE_MAP: dict[str, str] = {
    "int8": "integer",
    "int16": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint8": "integer",
    "uint16": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "float16": "decimal",
    "float32": "decimal",
    "float64": "decimal",
    "bool": "boolean",
    "object": "string",
    "string": "string",
    "category": "string",
    "datetime64[ns]": "timestamp",
    "datetime64[us]": "timestamp",
    "datetime64[ns, UTC]": "timestamp",
}


def infer_schema(df: pd.DataFrame) -> dict[str, str]:
    schema: dict[str, str] = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        schema[col] = DTYPE_MAP.get(dtype_str, "string")
    return schema
