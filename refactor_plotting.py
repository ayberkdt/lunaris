import re

f = 'analysis/monte_carlo/plotting.py'
with open(f, 'r', encoding='utf-8') as file:
    c = file.read()

# Remove old function definitions
c = re.sub(r'def _safe_float.*?return math\.nan\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_percent.*?return f.*?%\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_days.*?return f.*? d"\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_km.*?return f.*? km"\n', '', c, flags=re.DOTALL)

# Replace usage of old functions
c = c.replace('_format_', 'format_').replace('_safe_float', 'safe_float')

# Add imports
c = re.sub(
    r'(import numpy as np\n)',
    r'\1\nfrom analysis.formatting import safe_float, format_percent, format_days, format_km\n',
    c
)

with open(f, 'w', encoding='utf-8') as file:
    file.write(c)

print("Updated plotting.py")
