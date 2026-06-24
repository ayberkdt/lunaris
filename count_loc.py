import os
import sys

def count_loc(start_paths, exclude_dirs):
    total_files = 0
    total_lines = 0
    ext_stats = {}
    
    for start_path in start_paths:
        for root, dirs, files in os.walk(start_path):
            # Exclude directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ['.py', '.ts', '.tsx', '.js', '.jsx', '.html', '.css', '.md', '.json', '.yaml', '.yml', '.toml', '.c', '.cpp', '.h', '.hpp']:
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = sum(1 for _ in f)
                    total_files += 1
                    total_lines += lines
                    
                    if ext not in ext_stats:
                        ext_stats[ext] = {"files": 0, "lines": 0}
                    ext_stats[ext]["files"] += 1
                    ext_stats[ext]["lines"] += lines
                except Exception:
                    pass
                    
    return total_files, total_lines, ext_stats

exclude = {'node_modules', '__pycache__', 'outputs', 'data', 'venv', '.venv', 'build', 'dist', 'locks'}
paths = ['src', 'tests', 'validation', 'configs', 'docs']

files, lines, stats = count_loc(paths, exclude)
print(f"Total Files (source/docs/config): {files}")
print(f"Total Lines of Code: {lines}")
print("\nBreakdown by Extension:")
for ext, data in sorted(stats.items(), key=lambda x: x[1]['lines'], reverse=True):
    print(f"  {ext}: {data['files']} files, {data['lines']} lines")
