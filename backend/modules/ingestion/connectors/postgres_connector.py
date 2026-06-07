import pandas as pd
import psycopg2


class PostgresConnector:
    def __init__(self, config: dict):
        self.config = config

    def stream_table(self, query: str, chunk_size: int = 100_000):
        conn = psycopg2.connect(**self.config)
        cursor = conn.cursor(name="stream_cursor")
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]

        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            yield columns, rows

        cursor.close()
        conn.close()

    def read_all(self) -> pd.DataFrame:
        table = self.config.get("table")
        query = self.config.get("query") or f'SELECT * FROM "{table}"'
        config = {k: v for k, v in self.config.items() if k not in ("table", "query")}

        chunks = []
        for columns, rows in self.stream_table(query):
            chunks.append(pd.DataFrame(rows, columns=columns))

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
