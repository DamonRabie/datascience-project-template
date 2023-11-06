import configparser
from urllib.parse import quote

import pandas as pd
from sqlalchemy import create_engine

config = configparser.ConfigParser()
config.read(['configs/config.cfg', 'configs/config.dev.cfg'])


def create_sqlalchemy_engine_dotenv():
    """
    Read credentials from config file
    Looks for ['SQLSERVER'] key

    return: SQA engine
    """

    sqlserver_settings = config['SQLSERVER']

    SERVER = sqlserver_settings["SQL_SERVER"]
    USER = sqlserver_settings["SQL_USER"]
    PASS = sqlserver_settings["SQL_PASS"]
    DATABASE = sqlserver_settings["SQL_DATABASE"]

    engine = create_engine("mssql+pymssql://{}:{}@{}/{}".format(USER, quote(PASS), SERVER, DATABASE))
    return engine


def df_to_dw(save_dataframe: pd.DataFrame, schema: str, table_name: str, if_exists: str = 'append'):
    """
    save dataframe to sql server db
    :param save_dataframe: pandas dataframe to be saved
    :param schema: schema of database
    :param table_name: name of the designated table
    :param if_exists: sqlalchemy parameter. what to do if table already exists? append on default
    """
    sqa_engine = create_sqlalchemy_engine_dotenv()
    save_dataframe.to_sql(name=table_name, con=sqa_engine, schema=schema, if_exists=if_exists, index=False,
                          method='multi', chunksize=1000)


def fetch_query(query_str: str) -> pd.DataFrame:
    """
    fetch data from sql server db
    :param query_str: query string to run
    """
    sqa_engine = create_sqlalchemy_engine_dotenv()
    df = pd.read_sql_query(query_str, sqa_engine)
    return df
