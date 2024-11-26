import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion from data path.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize the data ingestion class."""
        self.data_path = data_path
        

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str):
    """
    Data ingestion from data path.
    Args:
        data_path: path to data csv file
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        return IngestData(data_path).get_data()
    except Exception as e:
        logging.error(f"Error while ingesting the data: {e}")
        raise e


# Pydantic is unable to generate a schema for a pandas.DataFrame object. 
# Pydantic does not natively support DataFrame as a type because it cannot validate or serialize it.
# Hence to pandas type annotations for now