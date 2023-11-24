## AryaXAI: ML Observability for mission-critical ‘AI’

AryaXAI is a full-stack ML Observability platform that integrates with your MLOPs platform to Explain, Monitor, Audit and Improve your ML models.

AryaXAI has multiple components to address the complex observability required for mission-critical ‘AI’. 

1. **ML Explainability:** AryaXAI offers diverse explainability options like- Bactrace(Specialized for deep learning models), SHAPE, Decision View, Observations (New way to correlate expert functioning vs model functioning) and Similar Cases (reference as explanations).
2. **ML Monitoring:** Monitor your models for drifts, performance & bias. The tool offers diverse options for drift (data/model) like - PSI, KL Divergence, Chi-square test, 
3. **Synthetic ‘AI’:** Deploy advanced synthetic ‘AI’ techniques like GPT-2 & GANs on your tabular data to generate high-quality synthetic datasets. Test the quality and privacy of these data sets using our Anonymity tests, column tests etc. 
4. **ML Risk policies:** Define advanced risk policies on your models. 
5. **AutoML:** AryaXAI also provides fully low-code and no-code options to build ML models on your data. For advanced users, it also provides more options to fine-tune it. 

AryaXAI also acts as a common workflow and provides insights acceptable by all stakeholders - Data Science, IT, Risk, Operations and compliance teams, making the rollout and maintenance of AI/ML models seamless and clutter-free.

### Quickstart:
Get started with AryaXAI with a few easy steps:

1. Sign up and log in to your new AryaXAI account.
2. After logging in, generate an Access Token for your user account.
3. Set the environment variable **XAI_ACCESS_TOKEN** with the generated value. 

Once you've completed these steps, you're all set! Now, you can easily log in and start using the AryaXAI SDK:

1. Log in by importing the "xai" object instance from the "arya_xai" package.
2. Call the "login" method. This method automatically takes the access token value from the "XAI_ACCESS_TOKEN" environment variable and stores the JWT in the object instance. This means that all your future SDK operations will be authorized automatically, making it simple and hassle-free!


```
from aryaxai import xai as aryaxai

## login() function authenticates user using token that can be generated in app.aryaxai.com/sdk


aryaxai.login()


Enter your Arya XAI Access Token: ··········
Authenticated successfully.
```

### Cookbook: 
In this section, you can review the examples of implementation of AryaXAI-SDK. 

1. [Full features overview of AryaXAI](https://colab.research.google.com/drive/1Dy5eL-FJVnFV0K5yOfGGVoAmiS_Icaz3?usp=sharing)
2. Using AryaXAI in Loan Underwriting (Coming Soon)

### Contribution guidelines:
At AryaXAI, we're passionate about open source and value community contributions! Explore our contribution guide for insights into the development workflow and AryaXAI library internals. For bug reports or feature requests, head to GitHub Issues or reach out to us at [support@aryaxai.com](support@aryaxai.com).