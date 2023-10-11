import os

__version__ = "0.0.1"
env = os.environ.get('XAI_ENV', 'local')

def get_version():
    """prints SDK version
    """
    return __version__

print(f'Welcome, you are using AryaXAI {get_version()} in {env} environment.')

#import core modules so that they can be imported directly
from arya_xai.core.xai import XAI

xai = XAI()