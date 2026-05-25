import re

f = 'analysis/reporting/manager.py'
with open(f, 'r', encoding='utf-8') as file:
    c = file.read()

# Remove old function definitions
c = re.sub(r'def _format_duration.*?return f.*?\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _safe_float.*?return float\("nan"\)\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_count.*?return "N/A"\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_percent.*?return f.*?%\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_days.*?return f.*? d"\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_km.*?return f.*? km"\n', '', c, flags=re.DOTALL)
c = re.sub(r'def _format_sci_or_na.*?return f.*?e"\n', '', c, flags=re.DOTALL)

# Replace usage of old functions
c = c.replace('_format_', 'format_').replace('_safe_float', 'safe_float')

# Add imports
c = re.sub(
    r'(import numpy as np\n)',
    r'\1\nfrom analysis.formatting import safe_float, format_duration, format_count, format_percent, format_days, format_km, format_sci_or_na\n',
    c
)

with open(f, 'w', encoding='utf-8') as file:
    file.write(c)

print("Updated manager.py")
