---
name: statistical-analysis
description: Guided statistical analysis with test selection and reporting. Use when you need help choosing appropriate tests for your data, assumption checking, power analysis, and APA-formatted results. Best for academic research reporting, test selection guidance. For implementing specific models programmatically use statsmodels.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

## Overview
Statistical analysis is a systematic process for testing hypotheses and quantifying relationships. Conduct hypothesis tests (t-test, ANOVA, chi-square), regression, correlation, and Bayesian analyses with assumption checks and APA reporting.

## When to Use This Skill
- Conducting statistical hypothesis tests (t-tests, ANOVA, chi-square)
- Performing regression or correlation analyses
- Running Bayesian statistical analyses
- Checking statistical assumptions and diagnostics
- Calculating effect sizes and conducting power analyses
- Reporting statistical results in APA format
- Analyzing Monte Carlo simulation output distributions

## Test Selection Guide

**Comparing Two Groups:**
- Independent, continuous, normal → Independent t-test
- Independent, continuous, non-normal → Mann-Whitney U test
- Paired, continuous, normal → Paired t-test
- Paired, continuous, non-normal → Wilcoxon signed-rank test
- Binary outcome → Chi-square or Fisher's exact test

**Comparing 3+ Groups:**
- Independent, continuous, normal → One-way ANOVA
- Independent, continuous, non-normal → Kruskal-Wallis test
- Paired, continuous, normal → Repeated measures ANOVA

**Relationships:**
- Two continuous variables → Pearson (normal) or Spearman correlation (non-normal)
- Continuous outcome with predictor(s) → Linear regression
- Binary outcome with predictor(s) → Logistic regression

## Assumption Checking — ALWAYS CHECK FIRST
```python
import scipy.stats as stats
import numpy as np

# Normality testing
stat, p = stats.shapiro(data)  # For n < 50
print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
# p > 0.05: normal distribution

# Homogeneity of variance
stat, p = stats.levene(group1, group2, group3)
print(f"Levene's test: F={stat:.4f}, p={p:.4f}")
# p > 0.05: equal variances

# Q-Q plot for normality
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(data, dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot')
plt.tight_layout()
```

**When Assumptions Are Violated:**
- Normality violated + n > 30/group → Proceed (robust due to CLT)
- Normality violated severely → Use non-parametric alternative
- Homogeneity violated (t-test) → Use Welch's t-test

## T-Test with Complete Reporting
```python
import pingouin as pg
import numpy as np

# Run independent t-test
result = pg.ttest(group_a, group_b, correction='auto')

t_stat = result['T'].values[0]
df = result['dof'].values[0]
p_value = result['p-val'].values[0]
cohens_d = result['cohen-d'].values[0]
ci_lower, ci_upper = result['CI95%'].values[0]

print(f"t({df:.0f}) = {t_stat:.2f}, p = {p_value:.3f}")
print(f"Cohen's d = {cohens_d:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}]")
```

## ANOVA with Post-Hoc Tests
```python
import pingouin as pg

aov = pg.anova(dv='score', between='group', data=df, detailed=True)
print(aov)

if aov['p-unc'].values[0] < 0.05:
    posthoc = pg.pairwise_tukey(dv='score', between='group', data=df)
    print(posthoc)

eta_squared = aov['np2'].values[0]  # Partial eta-squared
print(f"Partial η² = {eta_squared:.3f}")
```

## Linear Regression with Diagnostics
```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_const = sm.add_constant(X_predictors)
model = sm.OLS(y, X_const).fit()
print(model.summary())

# Check multicollinearity (VIF < 5 is acceptable)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)
```

## Power Analysis
```python
from statsmodels.stats.power import tt_ind_solve_power, FTestAnovaPower

# T-test: What n is needed to detect d = 0.5?
n_required = tt_ind_solve_power(
    effect_size=0.5, alpha=0.05, power=0.80,
    ratio=1.0, alternative='two-sided'
)
print(f"Required n per group: {n_required:.0f}")

# Sensitivity: With n=50, what effect could we detect?
detectable_d = tt_ind_solve_power(
    effect_size=None, nobs1=50, alpha=0.05,
    power=0.80, ratio=1.0, alternative='two-sided'
)
print(f"Study could detect d ≥ {detectable_d:.2f}")
```

## Effect Sizes Quick Reference
| Test | Effect Size | Small | Medium | Large |
|------|-------------|-------|--------|-------|
| T-test | Cohen's d | 0.20 | 0.50 | 0.80 |
| ANOVA | η²_p | 0.01 | 0.06 | 0.14 |
| Correlation | r | 0.10 | 0.30 | 0.50 |
| Regression | R² | 0.02 | 0.13 | 0.26 |

## APA Report Templates

### Independent T-Test
```
Group A (n = 48, M = 75.2, SD = 8.5) scored significantly higher than
Group B (n = 52, M = 68.3, SD = 9.2), t(98) = 3.82, p < .001, d = 0.77,
95% CI [0.36, 1.18], two-tailed.
```

### One-Way ANOVA
```
A one-way ANOVA revealed a significant main effect of treatment condition
on test scores, F(2, 147) = 8.45, p < .001, η²_p = .10. Post hoc
comparisons using Tukey's HSD indicated ...
```

## Best Practices
1. **Pre-register analyses** when possible
2. **Always check assumptions** before interpreting results
3. **Report effect sizes** with confidence intervals
4. **Report all planned analyses** including non-significant results
5. **Distinguish statistical from practical significance**
6. **Visualize data** before and after analysis
7. **Conduct sensitivity analyses** to assess robustness

## Installation
```bash
uv pip install scipy statsmodels pingouin pandas numpy
```
