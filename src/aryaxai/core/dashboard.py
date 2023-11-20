from typing import Any
from pydantic import BaseModel
import json
from IPython.display import IFrame, display

class Dashboard(BaseModel):
    config: dict
    url: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.print_config()
        self.plot()

    def plot(self, width: int='100%', height: int=800):
        """plot the dashboard by remote url

        Args:
            width (int, optional): _description_. Defaults to 100%.
            height (int, optional): _description_. Defaults to 650.
        """
        display(
            IFrame(src=f"{self.url}", width=width, height=height)
        )

    def get_config(self) -> dict:
        """
        get the dashboard config
        """
        return self.config

    def print_config(self):
        """
        pretty print the cdashboard config
        """
        config = {k: v for k, v in self.config.items() if v is not None}

        print("Using config: ", end='')
        print(json.dumps(config, indent=4))
        
    def __print__(self) -> str:
        return f"Dashboard(config='{self.config}')"

    def __str__(self) -> str:
        return self.__print__()

    def __repr__(self) -> str:
        return self.__print__()