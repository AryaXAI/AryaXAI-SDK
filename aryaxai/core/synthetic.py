from typing import List
from pydantic import BaseModel
import pandas as pd

class SyntheticDataTag(BaseModel):
    info: dict
    metadata: dict
    plot_data: List[dict]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_psi(self):
        """plot psi chart"""
        pass

    def view_info(self) -> pd.DataFrame:
        """pretty print the synthetic data tag info

        :return: DataFrame
        """
        info = {k: v for k, v in self.info.items() if v is not None}

        return pd.DataFrame(info, index=[0])

    def view_metadata(self) -> pd.DataFrame:
        """pretty print the synthetic data metadata
        
        :return: DataFrame
        """
        metadata = {k: v for k, v in self.metadata.items() if v is not None}

        return pd.DataFrame(metadata, index=[0])
        
    def __print__(self) -> str:
        return f"SyntheticDataTag({self.info})"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()
    
    
class SyntheticModel:
    pass