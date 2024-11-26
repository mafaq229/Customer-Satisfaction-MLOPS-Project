# from zenml.steps import BaseStep  # since this class is deprecated
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name: str = "LinearRegression"