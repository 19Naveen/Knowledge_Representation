import pandas as pd


class SnowflakeConnector:
    def __init__(self, config: dict):
        self.config = config

    def read_all(self) -> pd.DataFrame:
        raise NotImplementedError("Snowflake connector is not yet implemented")
