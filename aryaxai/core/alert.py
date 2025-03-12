from typing import List, Optional
from pydantic import BaseModel
import pandas as pd


class Alert(BaseModel):
    info: dict
    detailed_report: Optional[List[dict] | dict] = None
    not_used_features: Optional[List[dict]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def view_info(self) -> pd.DataFrame:
        """view the alert info

        :return: _description_
        """
        if not self.info:
            return "There was an error while executing the alert."

        return pd.DataFrame([self.info])

    def view_detailed_report(self) -> pd.DataFrame:
        """view the detailed report of alert

        :return: _description_
        """
        if not self.detailed_report:
            return "No detailed report found for the alert."

        if isinstance(self.detailed_report, list):
            detailed_report = [
                {
                    key: value
                    for key, value in feature.items()
                    if key != "current_small_hist" and key != "ref_small_hist"
                }
                for feature in self.detailed_report
            ]
        if isinstance(self.detailed_report, dict):
            detailed_report = [self.detailed_report]

        return pd.DataFrame(detailed_report)

    def view_not_used_features(self) -> pd.DataFrame:
        """view the not used features of alert

        :return: _description_
        """
        if not self.not_used_features:
            return "Not used features is empty."

        return pd.DataFrame(self.not_used_features)
