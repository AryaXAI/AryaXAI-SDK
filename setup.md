## SDK Development

1) Set environment variable XAI_ENV=local on your machine for local development.

2) Update build tools and Build SDK
```
pip install --upgrade build && python -m build
```

3) Install locally
    - Go to src directory.
    - Install package in editable mode. Relaunch python shell after you make local changes to see latest changes.
    ```
    pip install -e .
    ```

### SDK release on testing environment
1) Checkout "testing" branch and merge your changes.

2) Update package version manually in setup.py file and push changes.

3) Set environment variable XAI_ENV=testing on your machine and make sure you restart machine or code editor so that testing environment is picked up by package code.

4) Release to testpypi
```
py -m build
py -m twine upload --repository testpypi dist/*
```

### SDK release on production environment
- SDK is auto released on production environment when changes are pushed on "main" branch.

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