#create a dashboard class that have url as param and plot method that displays the iframe
from pydantic import BaseModel
from IPython.display import IFrame, display

class Dashboard(BaseModel):
    url: str

    def plot(self, width: int=800, height: int=650):
        """plot the dashboard by remote url

        Args:
            width (int, optional): _description_. Defaults to 800.
            height (int, optional): _description_. Defaults to 650.
        """
        display(
            IFrame(src=f"{self.url}", width=width, height=height)
        )