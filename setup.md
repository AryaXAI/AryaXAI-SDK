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
import arya_xai
arya_xai.get_version()
```

## Publish SDK

### Set environment variable
> XAI_ENV=prod

### Checkout main
```
git checkout origin/main
```

### Build and Publish SDK
```
gh release create --repo AryaXAI/arya-xai-sdk --generate-notes <Your-Version>
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