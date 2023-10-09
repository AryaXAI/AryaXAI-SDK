import os
from pydantic import BaseModel

class xai(BaseModel):
    def login(self, api_key=None):
        api_key = api_key or os.environ.get('ARYAXAI_API_KEY', None)
        
        if not api_key:
            raise ValueError("Either set ARYAXAI_API_KEY or pass the API key in XAIBase class.")
        
        print('Authenticating API Key...')

        # TODO: Authenticate API key and get authtoken
