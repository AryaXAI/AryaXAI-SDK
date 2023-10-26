import os

env = os.environ.get('XAI_ENV', 'prod')
logger_on = os.environ.get('DEBUG', False)

if logger_on:
    print(f'Welcome, you are using AryaXAI in {env} environment.')

#import core modules so that they can be imported directly
from aryaxai.core.xai import XAI

xai = XAI()