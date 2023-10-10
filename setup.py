import os
import sys
from setuptools import find_packages, Command, setup

if sys.version_info < (3, 0):
    sys.exit('Sorry, Python < 3.0 is not supported.')
    
def set_environment():
    env = 'devv'
    
    if "--env" in sys.argv:
        index = sys.argv.index("--env")
        if index < len(sys.argv) - 1:
            env = sys.argv[index + 1]
        
    os.environ['XAI_ENV'] = env
    

environment_name = set_environment()
print(os.environ.get('XAI_ENV')) 
    
# AryaXAI SDK dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name="arya-xai",
    version="0.0.1",
    author="AryaXAI",
    description="AryaXAI - Interact with AryaXAI services",
    long_description='AryaXAI is a full stack ML Observability tool for mission-critical AI functions. Designed by Arya.ai, it is aimed to deliver much required common platform between stakeholders and deliver AI transparency, trust and auditability.',
    long_description_content_type="text/plain",
    keywords="arya-xai, ML observability",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.0',
    install_requires=requirements
)