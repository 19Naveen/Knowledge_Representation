import pandas as pd

from modules.ingestion.storage.minio_client import download_staging_file


class CsvConnector:
    def __init__(self, staging_path: str):
        self.staging_path = staging_path

    def read(self) -> pd.DataFrame:
        local_path = f"/tmp/{self.staging_path.replace('/', '_')}"
        download_staging_file(self.staging_path, local_path)
        return pd.read_csv(local_path)


class XlsxConnector:
    def __init__(self, staging_path: str):
        self.staging_path = staging_path

    def read(self) -> pd.DataFrame:
        local_path = f"/tmp/{self.staging_path.replace('/', '_')}"
        download_staging_file(self.staging_path, local_path)
        return pd.read_excel(local_path, engine="openpyxl")
