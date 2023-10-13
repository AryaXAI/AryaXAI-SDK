A python client for [AryaXAI](https://xai.arya.ai).

AryaXAI is a full stack ML Observability tool for mission-critical AI functions. Designed by Arya.ai, it is aimed to deliver much required common platform between stakeholders and deliver AI transparency, trust and auditability.

### Installation
```
pip install arya-xai
```

### Getting Started
```
from arya_xai import xai

xai.login()

workspaces = xai.get_workspaces()
```