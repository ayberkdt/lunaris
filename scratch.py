import re

paths = ['tests/test_st_lrps.py', 'tests/test_st_lrps_fixes.py']
for p in paths:
    with open(p, 'r', encoding='utf8') as f:
        text = f.read()

    # The cfg dict usually has "altitude_max_km": 500.0,
    # Let's just insert the dataset block right after altitude_max_km": 500.0,
    pattern = r'("altitude_max_km": \d+\.\d+,?)'
    replacement = r'\1 "dataset": {"target_mode": "residual", "degree_min": 10, "degree_max": 60, "altitude_min_km": 100.0, "altitude_max_km": 500.0},'
    
    new_text = re.sub(pattern, replacement, text)
    
    # Let's also handle {"resolved_mu_si": 4.902e12, "resolved_a_sign": 1.0, "resolved_r_ref_m": 1.737e6, "degree_min": -1}
    # which doesn't have altitude_max_km
    pattern2 = r'("degree_min": -1)(?!, "altitude)'
    replacement2 = r'\1, "dataset": {"target_mode": "residual", "degree_min": 10, "degree_max": 60, "altitude_min_km": 100.0, "altitude_max_km": 500.0}'
    new_text = re.sub(pattern2, replacement2, new_text)

    with open(p, 'w', encoding='utf8') as f:
        f.write(new_text)
