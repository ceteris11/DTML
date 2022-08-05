
from sqlalchemy import create_engine
from contextlib import contextmanager
import pandas as pd
import json
import os


class DbConnector:
    def __init__(self, filename):
        # init var
        self.filename = filename
        self.config = None
        self.engine = None

        # set config
        self.set_config()
        self.set_engine()

    def get_config(self, k=None):
        if k is None:
            return self.config
        else:
            return self.config[k]

    def set_config(self):
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/{self.filename}') as f:
            self.config = json.load(f)

    def set_engine(self):
        # sqlalchemy 설정
        self.engine = create_engine(
            f'mysql+pymysql://{self.get_config("user")}:{self.get_config("pw")}@{self.get_config("url")}'
            f':{self.get_config("port")}'
            f'/{self.get_config("schema")}'
            f'?autocommit=true', pool_size=20, pool_recycle=20, max_overflow=100)

    def get_conn(self):
        return self.engine.connect()


class DbManager:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "_init"):
            self.db_dict = {'galaxynet': DbConnector('galaxynet_config.json')}
            self.default_db = 'galaxynet'

            # set init flag
            cls._init = True

    def get_config(self, db_key=None, k=None):
        if db_key is None:
            db_key = self.default_db
        return self.db_dict[db_key].get_config(k)

    def get_conn(self, db_key=None):
        if db_key is None:
            db_key = self.default_db
        return self.db_dict[db_key].get_conn()

    @contextmanager
    def conn_context(self, db_key=None):
        if db_key is None:
            db_key = self.default_db
        conn = self.get_conn(db_key)
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except Exception as e:
            trans.rollback()
            raise e
        finally:
            conn.close()


if __name__ == '__main__':
    # get sql query result
    with DbManager().conn_context() as conn:
        pd.read_sql('select * from stock_daily_technical where date = 20220218', conn)
