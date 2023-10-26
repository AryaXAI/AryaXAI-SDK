## SDK Development

### Set environment
- Set XAI_ENV=local on your machine for local development.
- Set XAI_API_KEY={Your-API-key}

### Update build tools
```
pip install --upgrade build
```

### Build SDK
```
python -m build
```

### Install locally
Install SDK by running below command in root directory. -e option watches for changes in SDK, so you dont have to build SDK again. Just relaunch python shell to see latest changes.
```
pip install -e .
```

### Quicktest
Launch Python shell and run below commands to in python shell to quicktest SDK,
```
from aryaxai import xai
xai.get_workspaces()
```
```

## Publish SDK

### Set environment variable
> XAI_ENV=prod

### Checkout main
```
git checkout origin/main
```

### Build and Publish SDK
1) Go to github.com and launch release workflow for main branch.
or
2) Manual release
```
py -m build
py -m twine upload dist/*
```

## Documentation
Go to root directory and run below command. It updates documentation schema files (.rst files)
```
sphinx-apidoc -o docs aryaxai/
```

Go to docs directory and run below command (generates html template from .rst files)
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