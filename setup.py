import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 0):
    sys.exit('Sorry, Python < 3.0 is not supported.')
    
# AryaXAI SDK dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
# Add readme as long description
with open('readme.md') as fr:
    long_description = fr.read()
    

setup(
    name="arya-xai",
    version="0.0.1",
    author="AryaXAI",
    description="AryaXAI - Interact with AryaXAI services",
    long_description=long_description,
    long_description_content_type="text/markdown",
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