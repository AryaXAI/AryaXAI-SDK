## SDK Development

### Set environment
- Set XAI_ENV=local on your machine for local development.
- Set XAI_API_KEY={Your-API-key}

### Update build tools
```
pip install setuptools wheel --upgrade
```

### Build SDK
```
python setup.py sdist bdist_wheel
```

### Install locally
Install SDK by running below command in root directory. -e option watches for changes in SDK, so you dont have to build SDK again. Just relaunch python shell to see latest changes.
```
pip install -e .
```

### Quicktest
Launch Python shell and run below commands to in python shell to quicktest SDK,
```
import arya_xai
arya_xai.get_version()
```

## Publish SDK

### Set environment variable
> XAI_ENV=prod

### Install twine
```
pip install twine
```

### Build SDK
```
python setup.py sdist bdist_wheel
```

### Verify build
Check your distribution files for errors,
```
twine check dist/*
```

### Set crendentials
Ensure you have PyPi (https://pypi.org/) user account and you are collaborator to arya-xai project. Generate API token from your account and put it in file .pypirc in your home($HOME/.pypirc) or user(C:\Users\<your username>) directory as given below,
```
[pypi]
username = __token__
password = <Your-api-token>
```

### Publish SDK
```
twine upload dist/*
```

## Documentation
Go to docs directory,
```
cd docs
```
Clean exisiting documentation,
```
make clean html
```
Generate documentation,
```
make html
```