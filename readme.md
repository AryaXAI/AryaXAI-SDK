A python client for [AryaXAI](https://xai.arya.ai).

### Installation
```
pip install aryaxai
```

### Getting Started
```
from aryaxai import xai

xai.login()

# list of workspaces
workspaces = xai.get_workspaces()

# projects from first workspace
projects = workspaces[0].get_projects()
```

### Example Notebook
Refer to the [notebook](https://colab.research.google.com/drive/1Dy5eL-FJVnFV0K5yOfGGVoAmiS_Icaz3?usp=sharing) for detailed examples.