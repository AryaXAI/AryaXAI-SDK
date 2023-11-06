import os
from pydantic import BaseModel
from dotenv import load_dotenv

class Environment(BaseModel):
    """
    Environment class to load current environment
    """
    debug: bool = False
    XAI_ENV: str = os.getenv("XAI_ENV", 'prod')
    
    def __init__(self):
        super().__init__()

        self.load_environment()
        
    def load_environment(self):
        """
        load current environment config
        """
        env_file = f'.env.{self.XAI_ENV}'
        
        BASEDIR = os.path.abspath(os.path.dirname(__file__))
        load_dotenv(os.path.join(BASEDIR, 'config', env_file))
   
        logger_on = self.get_debug()

        if logger_on:
            self.debug = logger_on
            print(f'Connected to: {self.XAI_ENV} environment')
        
    def get_base_url(self) -> str:
        """get base url of XAI platform

        :return: base url
        """
        return os.getenv("XAI_API_URL", "https://api-m.aryaxai.com")

    def get_debug(self) -> bool:
        """get debug flag

        :return: debug flag
        """
        return bool(os.getenv("DEBUG", False))
