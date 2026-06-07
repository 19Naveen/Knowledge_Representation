import pandas as pd


class MySQLConnector:
    def __init__(self, config: dict):
        self.config = config

    def read_all(self) -> pd.DataFrame:
        raise NotImplementedError("MySQL connector is not yet implemented")
