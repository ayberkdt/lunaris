---
name: pymc
description: Bayesian modeling with PyMC. Build hierarchical models, MCMC (NUTS), variational inference, LOO/WAIC comparison, posterior checks, for probabilistic programming and inference.
license: Apache License, Version 2.0
metadata:
    skill-author: K-Dense Inc.
---

## Overview
PyMC is a Python library for Bayesian modeling and probabilistic programming. Build, fit, validate, and compare Bayesian models using PyMC's modern API (version 5.x+), including hierarchical models, MCMC sampling (NUTS), variational inference, and model comparison (LOO, WAIC).

## When to Use This Skill
- Building Bayesian models (linear/logistic regression, hierarchical models, time series)
- Performing MCMC sampling or variational inference
- Conducting prior/posterior predictive checks
- Implementing uncertainty quantification (e.g., Monte Carlo orbit ensemble analysis)
- Diagnosing sampling issues (divergences, convergence, ESS)
- Comparing multiple models using information criteria (LOO, WAIC)

## Standard Bayesian Workflow

### 1. Data Preparation
```python
import pymc as pm
import arviz as az
import numpy as np

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize predictors
```

### 2. Model Building
```python
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=n_predictors)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha + pm.math.dot(X_scaled, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

### 3. Prior Predictive Check
```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=42)
az.plot_ppc(prior_pred, group='prior')
```

### 4. Fit Model
```python
with model:
    idata = pm.sample(draws=2000, tune=1000, chains=4,
                      target_accept=0.9, random_seed=42,
                      idata_kwargs={'log_likelihood': True})
```

### 5. Diagnostics
```python
print(az.summary(idata))  # Check R-hat < 1.01, ESS > 400
az.plot_posterior(idata)
```

### 6. Posterior Predictive Check
```python
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
az.plot_ppc(idata)
```

### 7. Make Predictions
```python
with model:
    pm.set_data({'X': X_new_scaled})
    post_pred = pm.sample_posterior_predictive(idata.posterior)
y_pred_mean = post_pred.posterior_predictive['y_obs'].mean(dim=['chain', 'draw'])
y_pred_hdi = az.hdi(post_pred.posterior_predictive, var_names=['y_obs'])
```

## Hierarchical Models (non-centered parameterization)
```python
with pm.Model(coords={'groups': group_names}) as hierarchical_model:
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, dims='groups')
    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset, dims='groups')
    mu = alpha[group_idx]
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)
```

## Model Comparison
```python
comparison = az.compare({'Model1': idata1, 'Model2': idata2}, ic='loo')
# Δloo < 2: models similar; Δloo > 10: strong evidence for better model
```

## Prior Selection
- Scale params: `pm.HalfNormal('sigma', sigma=1)`
- Unbounded: `pm.Normal('theta', mu=0, sigma=1)`
- Robust: `pm.StudentT('theta', nu=3, mu=0, sigma=1)`
- Positive: `pm.LogNormal('theta', mu=0, sigma=1)`

## Variational Inference (fast exploration)
```python
with model:
    approx = pm.fit(n=20000, method='advi')
    idata = pm.sample(start=approx.sample(return_inferencedata=False)[0])
```

## Troubleshooting
- **Divergences** → Increase `target_accept=0.95`, use non-centered parameterization
- **Low ESS** → More `draws`, reparameterize
- **High R-hat** → Longer chains, check multimodality

## Installation
```bash
uv pip install pymc arviz
```
