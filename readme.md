## Arya XAI

A python package to interact with Arya XAI services.

### Update build tools
> pip install --user setuptools wheel --upgrade

### Build package
> python setup.py sdist bdist_wheel

### Run quick test locally
1. Run below command to install package locally,
> pip install -e .

2. Go to python shell,
> Python

In python shell, run below statements to do quick test of package
```
import arya_xai
arya_xai.test()
```