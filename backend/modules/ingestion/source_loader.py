"""Load an ingestion job's source into a DataFrame.

Shared by the Celery pipeline (full load) and the staged-preview endpoint (sampled load).
Connectors are imported lazily so importing this module stays cheap.
"""

from modules.ingestion.enums import SourceType


def load_source(job, nrows: int | None = None):
    """Read the job's staged file / connected table into a pandas DataFrame.

    ``nrows`` bounds the result for previews (applied after read for v1 — file connectors
    read the whole staged object, then we slice). ``None`` returns the full dataset.
    """
    from modules.ingestion.connectors.csv import CsvConnector, XlsxConnector
    from modules.ingestion.connectors.parquet import ParquetConnector
    from modules.ingestion.connectors.postgres_connector import PostgresConnector
    from modules.ingestion.connectors.snowflake_connector import SnowflakeConnector
    from modules.ingestion.connectors.mysql_connector import MySQLConnector
    from modules.ingestion.connectors.mssql_connector import MSSQLConnector

    source = job.source_type
    if source == SourceType.CSV:
        df = CsvConnector(job.staging_path).read()
    elif source == SourceType.XLSX:
        df = XlsxConnector(job.staging_path).read()
    elif source == SourceType.PARQUET:
        df = ParquetConnector(job.staging_path).read()
    elif source == SourceType.POSTGRES:
        df = PostgresConnector(job.source_config).read_all()
    elif source == SourceType.SNOWFLAKE:
        df = SnowflakeConnector(job.source_config).read_all()
    elif source == SourceType.MYSQL:
        df = MySQLConnector(job.source_config).read_all()
    elif source == SourceType.MSSQL:
        df = MSSQLConnector(job.source_config).read_all()
    else:
        raise ValueError(f"Unknown source type: {source}")

    if nrows is not None:
        df = df.head(nrows)
    return df
