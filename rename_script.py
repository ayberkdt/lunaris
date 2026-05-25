import os
import re

dirs = ['analysis', 'validation', 'visualization']
targets = {'LUNAR_SIMULATION': 'ST_LRPS', 'LunarSim': 'ST_LRPS', 'LUNARSIM_': 'STLRPS_'}
pattern = re.compile('|'.join(re.escape(k) for k in targets.keys()))

files_to_update = []
for d in dirs:
    if not os.path.exists(d):
        continue
    for r, _, files in os.walk(d):
        if '__pycache__' in r:
            continue
        for f in files:
            if f.endswith('.py'):
                files_to_update.append(os.path.join(r, f))

for p in files_to_update:
    with open(p, 'r', encoding='utf-8') as file:
        content = file.read()
    
    new_content = pattern.sub(lambda m: targets[m.group(0)], content)
    
    if new_content != content:
        with open(p, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Updated {p}")
