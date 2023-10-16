A python client for [AryaXAI](https://xai.arya.ai).

### Installation
```
pip install arya-xai
```

### Getting Started
```
from arya_xai import xai

xai.login()

# list of workspaces
workspaces = xai.get_workspaces()

# projects from first workspace
projects = workspaces[0].get_projects()
```