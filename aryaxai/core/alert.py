from typing import List
from pydantic import BaseModel
import pandas as pd

class Alert(BaseModel):
    info: dict
    detailed_report: List[dict]
    not_used_features: List[dict]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # print(self.view_info())
        # print(self.view_detailed_report())
        # print(self.view_not_used_features())
        
    def view_info(self) -> pd.DataFrame:
        """view the alert info

        :return: _description_
        """
        if not self.info:
            return 'There was an error while executing the alert.'

        return pd.DataFrame([self.info])
        
    def view_detailed_report(self) -> pd.DataFrame:
        """view the detailed report of alert

        :return: _description_
        """
        if not self.detailed_report:
            return 'There was an error while executing the alert.'

        detailed_report = [{key: value for key, value in feature.items() if key != 'current_small_hist' and key != 'ref_small_hist'} for feature in self.detailed_report]

        return pd.DataFrame(detailed_report)

    def view_not_used_features(self) -> pd.DataFrame:
        """view the not used features of alert

        :return: _description_
        """
        if not self.not_used_features:
            return 'Not used features is empty.'

        return pd.DataFrame(self.not_used_features)
        
    def __print__(self) -> str:
        return self.view_info()

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()