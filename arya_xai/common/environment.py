

import os
from pydantic import BaseModel
from dotenv import load_dotenv

class Environment(BaseModel):
    """Environment class to load current environment
    """
    XAI_ENV: str
    
    def __init__(self):
        self.XAI_ENV = os.getenv("XAI_ENV", 'local')
        self.load_environment()
        
    def load_environment(self):
        """load current environment config
        """
        env_file = f'./config/.env.{self.XAI_ENV}'
        load_dotenv(env_file)
        
    def get_base_url(self) -> str:
        """get base url of XAI platform

        :return: base url
        """
        return os.getenv("BASE_URL", "default_base_url")
