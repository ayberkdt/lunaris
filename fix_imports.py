import os
import re

file_to_module = {
    "data_files_page": "pages",
    "force_models_page": "pages",
    "live_telemetry_page": "pages",
    "mission_propagation_page": "pages",
    "monte_carlo_page": "pages",
    "orbit_config_page": "pages",
    "result_exports_page": "pages",
    "monte_carlo_analysis_panel": "components",
    "showcase_embed": "components",
    "command_builder": "core",
    "gravity_artifact_utils": "core",
    "preflight_validation": "core",
    "session_persistence": "core",
    "solver_policy": "core",
    "surrogate_artifacts": "core",
    "ui_commons": "core"
}

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content
    
    # Replace relative imports like `from .ui_commons import X`
    for mod_name, folder in file_to_module.items():
        # Match `from .mod_name import `
        pattern = r"from\s+\." + mod_name + r"\s+import\s+"
        replacement = f"from lunaris.ui.{folder}.{mod_name} import "
        content = re.sub(pattern, replacement, content)
        
        # Match `from . import mod_name`
        pattern2 = r"from\s+\.\s+import\s+" + mod_name + r"\b"
        replacement2 = f"from lunaris.ui.{folder} import {mod_name}"
        content = re.sub(pattern2, replacement2, content)
        
        # Test imports: `lunaris.ui.widgets` -> `lunaris.ui.folder`
        content = content.replace(f"lunaris.ui.widgets.{mod_name}", f"lunaris.ui.{folder}.{mod_name}")
        content = content.replace(f"lunaris.ui.widgets import {mod_name}", f"lunaris.ui.{folder} import {mod_name}")

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for d in ["src", "tests"]:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                fix_file(os.path.join(root, file))

print("Import fix complete.")
