import os
from pathlib import Path

file_to_module = {
    "data_files_page.py": "pages",
    "force_models_page.py": "pages",
    "live_telemetry_page.py": "pages",
    "mission_propagation_page.py": "pages",
    "monte_carlo_page.py": "pages",
    "orbit_config_page.py": "pages",
    "result_exports_page.py": "pages",
    "monte_carlo_analysis_panel.py": "components",
    "showcase_embed.py": "components",
    "command_builder.py": "core",
    "gravity_artifact_utils.py": "core",
    "preflight_validation.py": "core",
    "session_persistence.py": "core",
    "solver_policy.py": "core",
    "surrogate_artifacts.py": "core",
    "ui_commons.py": "core"
}

src_dir = Path("src/lunaris/ui/widgets")
dest_base = Path("src/lunaris/ui")

# Move files
for file_name, folder in file_to_module.items():
    old_path = src_dir / file_name
    new_path = dest_base / folder / file_name
    if old_path.exists():
        old_path.rename(new_path)

# Create init files
(dest_base / "pages" / "__init__.py").touch(exist_ok=True)
(dest_base / "components" / "__init__.py").touch(exist_ok=True)
(dest_base / "core" / "__init__.py").touch(exist_ok=True)

# Function to replace imports
def update_imports(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return
        
    original = content
    for file_name, folder in file_to_module.items():
        mod_name = file_name[:-3]
        old_import = f"lunaris.ui.widgets.{mod_name}"
        new_import = f"lunaris.ui.{folder}.{mod_name}"
        content = content.replace(old_import, new_import)
        
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {filepath}")

# Update imports in src and tests
for d in ["src", "tests"]:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                update_imports(os.path.join(root, file))

print("Done")
