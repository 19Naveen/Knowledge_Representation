import enum


class JobStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCESS = "SUCCESS"


class SourceType(enum.Enum):
    CSV = "csv"
    XLSX = "xlsx"
    PARQUET = "parquet"
    POSTGRES = "postgres"
    SNOWFLAKE = "snowflake"
    MYSQL = "mysql"
    MSSQL = "mssql"


class TransformType(enum.Enum):
    MAP = "map"
    DROP = "drop"
    CAST = "cast"
    FILTER = "filter"
    FILLNA = "fillna"
