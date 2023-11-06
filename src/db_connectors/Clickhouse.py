import configparser

import pandas as pd
from clickhouse_driver import Client

config = configparser.ConfigParser()
config.read(['configs/config.cfg', 'configs/config.dev.cfg'])


def create_clickhouse_engine_dotenv():
    """
    Read credentials from config file
    Looks for ['CLICKHOUSE'] key

    return: clickhouse driver
    """

    clickhouse_settings = config['CLICKHOUSE']

    SERVER = clickhouse_settings["CK_HOST"]
    PORT = clickhouse_settings["CK_PORT"]
    USER = clickhouse_settings["CK_USER"]
    PASS = clickhouse_settings["CK_PASSWORD"]
    DATABASE = clickhouse_settings["CK_DATABASE"]

    return Client.from_url(f'clickhouse://{USER}:{PASS}@{SERVER}:{PORT}/{DATABASE}?use_numpy=True')


def run_query(query_str: str) -> pd.DataFrame:
    client = create_clickhouse_engine_dotenv()
    df = client.query_dataframe(query_str)
    return df
