import os

__version__ = "0.0.1"
env = os.environ.get('XAI_ENV', 'dev')

print(f'Welcome, you are using AryaXAI {__version__} in {env} environment.')

from arya_xai.core.xai import xai