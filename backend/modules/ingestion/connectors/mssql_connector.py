import pandas as pd


class MSSQLConnector:
    def __init__(self, config: dict):
        self.config = config

    def read_all(self) -> pd.DataFrame:
        raise NotImplementedError("SQL Server connector is not yet implemented")
