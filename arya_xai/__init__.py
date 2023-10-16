import os

env = os.environ.get('XAI_ENV', 'local')

print(f'Welcome, you are using AryaXAI in {env} environment.')

#import core modules so that they can be imported directly
from arya_xai.core.xai import XAI

xai = XAI()