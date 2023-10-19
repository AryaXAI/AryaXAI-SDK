import os
from pydantic import BaseModel
from dotenv import load_dotenv

class Environment(BaseModel):
    """Environment class to load current environment
    """
    XAI_ENV: str = os.getenv("XAI_ENV", 'testing')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_environment()
        
    def load_environment(self):
        """load current environment config
        """
        env_file = f'.env.{self.XAI_ENV}'
        
        BASEDIR = os.path.abspath(os.path.dirname(__file__))
        load_dotenv(os.path.join(BASEDIR, 'config', env_file))
        
    def get_base_url(self) -> str:
        """get base url of XAI platform

        :return: base url
        """
        return os.getenv("XAI_API_URL")
