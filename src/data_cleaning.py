import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):  
    """
    Abstract class defining strategy for handling data (Strategy Pattern)
    """
    
    @abstractmethod  # abstract class i.e. we can overwrite this method (just a blue print)
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    

class DataPreProcessStrategy(DataStrategy):
    """_summary_
    Strategy for preprocessing data
    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop(columns=[
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"
            ])
            
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            
            # Only selecting column with numerical values
            data = data.select_dtypes(include=[np.number])
            
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(columns=cols_to_drop)
            
            return data
            
        except Exception as e:
            logging.error(f"Error in  preprocessing data: {e}") # F strings are better for performance and readibility
            # logging.error("Error in  preprocessing data: {}".format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):
    """_summary_
    Strategy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide the data into train and test
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning: # 
    """
    Class for cleaning data which process the data and divides it 
    into train and test (makes use of strategies defined above)
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
