---
name: scientific-visualization
description: Meta-skill for publication-ready figures. Use when creating journal submission figures requiring multi-panel layouts, significance annotations, error bars, colorblind-safe palettes, and specific journal formatting (Nature, Science, Cell). Orchestrates matplotlib/seaborn/plotly with publication styles.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

## Overview
Scientific visualization transforms data into clear, accurate figures for publication. Create journal-ready plots with multi-panel layouts, error bars, significance markers, and colorblind-safe palettes. Export as PDF/EPS/TIFF using matplotlib, seaborn, and plotly for manuscripts.

## When to Use This Skill
- Creating plots or visualizations for scientific manuscripts
- Preparing figures for journal submission
- Ensuring figures are colorblind-friendly and accessible
- Making multi-panel figures with consistent styling
- Exporting figures at correct resolution and format

## Basic Publication-Quality Figure
```python
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7

# Single column width for Nature: 3.5 inches (89 mm)
fig, ax = plt.subplots(figsize=(3.5, 2.5))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Amplitude (mV)')
ax.legend(frameon=False)

# Remove unnecessary spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('figure1.pdf', bbox_inches='tight')
plt.savefig('figure1.png', dpi=300, bbox_inches='tight')
```

## Colorblind Accessibility — ALWAYS USE
```python
# Okabe-Ito palette (distinguishable by all types of color blindness)
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=okabe_ito)

# For heatmaps/continuous data — use perceptually uniform colormaps
# GOOD: viridis, plasma, cividis
# BAD: jet, rainbow
# Diverging (OK): RdBu, PuOr, BrBG (NOT red-green)
```

## Multi-Panel Figures
```python
from string import ascii_uppercase

fig = plt.figure(figsize=(7, 4))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Add bold panel labels (A, B, C, D)
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.text(-0.15, 1.05, ascii_uppercase[i], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
```

## Journal-Specific Figure Widths
| Journal | Single column | Double column |
|---------|--------------|---------------|
| Nature  | 89 mm (3.5")  | 183 mm (7.2") |
| Science | 55 mm (2.2")  | 175 mm (6.9") |
| Cell    | 85 mm (3.3")  | 178 mm (7.0") |

## Statistical Rigor with Error Bars
```python
import seaborn as sns

# Always show uncertainty
fig, ax = plt.subplots(figsize=(3.5, 3))
sns.boxplot(data=df, x='treatment', y='response', palette='Set2', ax=ax)
sns.stripplot(data=df, x='treatment', y='response',
              color='black', alpha=0.3, size=3, ax=ax)

# Significance markers
ax.text(1.5, max_y * 1.1, '***', ha='center', fontsize=8)

# Error bar with errorbar
ax.errorbar(x, means, yerr=sems, fmt='o', capsize=3)
```

## Using Seaborn for Publication Plots
```python
import seaborn as sns
sns.set_theme(style='ticks', context='paper', font_scale=1.1)
sns.set_palette('colorblind')  # Colorblind-safe default

fig, ax = plt.subplots(figsize=(3.5, 2.5))
sns.lineplot(data=timeseries, x='time', y='measurement',
             hue='treatment', errorbar=('ci', 95),
             markers=True, ax=ax)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Measurement (AU)')
sns.despine()  # Remove top and right spines
```

## Seaborn Multi-Panel with Facets
```python
g = sns.relplot(data=df, x='dose', y='response',
                hue='treatment', col='cell_line',
                kind='line', height=2.5, aspect=1.2,
                errorbar=('ci', 95), markers=True)
g.set_axis_labels('Dose (μM)', 'Response (AU)')
g.set_titles('{col_name}')
sns.despine()
plt.savefig('figure_facets.pdf', bbox_inches='tight')
```

## Correlation Heatmap
```python
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(figsize=(5, 4))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=1, cbar_kws={'shrink': 0.8}, ax=ax)
plt.tight_layout()
```

## Publication Checklist
Before submitting figures, verify:
- [ ] Resolution meets journal requirements (300+ DPI)
- [ ] File format is correct (vector for plots, TIFF/PNG for images)
- [ ] Figure size matches journal specifications
- [ ] All text readable at final size (≥6 pt)
- [ ] Colors are colorblind-friendly (use Okabe-Ito or viridis)
- [ ] Figure works in grayscale
- [ ] All axes labeled with units
- [ ] Error bars present with definition in caption
- [ ] Panel labels present (A, B, C...) and consistent
- [ ] No chart junk or 3D effects
- [ ] Statistical significance clearly marked
- [ ] Legend is clear and complete

## Common Pitfalls to Avoid
1. **Font too small**: Text unreadable when printed at final size
2. **JPEG format**: Never use JPEG for graphs/plots (creates artifacts)
3. **Red-green colors**: ~8% of males cannot distinguish
4. **Low resolution**: Pixelated figures in publication
5. **Missing units**: Always label axes with units
6. **3D effects**: Distorts perception, avoid completely
7. **No error bars**: Always show uncertainty
