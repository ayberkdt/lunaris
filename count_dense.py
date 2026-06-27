import ast
import os
import tokenize
from io import BytesIO


def count_dense_python(start_paths, exclude_dirs):
    total_files = 0
    total_raw_lines = 0
    total_dense_lines = 0
    total_comments = 0
    total_empty = 0
    total_docstrings = 0

    for start_path in start_paths:
        for root, dirs, files in os.walk(start_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for file in files:
                if not file.endswith('.py'):
                    continue

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        source_bytes = f.read()

                    source_str = source_bytes.decode('utf-8')
                    raw_lines = source_str.split('\n')
                    total_raw_lines += len(raw_lines)
                    total_files += 1

                    # Count empty lines
                    empty = sum(1 for line in raw_lines if not line.strip())
                    total_empty += empty

                    # Use tokenize to find comments and docstrings accurately
                    tokens = tokenize.tokenize(BytesIO(source_bytes).readline)

                    comment_lines = set()
                    docstring_lines = set()

                    for tok in tokens:
                        if tok.type == tokenize.COMMENT:
                            # Mark the lines covered by the comment
                            for line_num in range(tok.start[0], tok.end[0] + 1):
                                comment_lines.add(line_num)
                        elif tok.type == tokenize.STRING:
                            # Heuristic: if a string is the only thing on a line or it's a multiline string
                            # that acts as a docstring (we could use AST for perfect docstrings, but tokenize is good for all loose strings)
                            pass

                    # Let's use AST for docstrings
                    try:
                        tree = ast.parse(source_str)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                                docstring = ast.get_docstring(node, clean=False)
                                if docstring and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                                    doc_node = node.body[0]
                                    for line_num in range(doc_node.lineno, doc_node.end_lineno + 1):
                                        docstring_lines.add(line_num)
                    except Exception:
                        pass

                    total_comments += len(comment_lines)
                    total_docstrings += len(docstring_lines)

                    # Dense lines = raw - empty - comments - docstrings
                    # Need to be careful not to double count if a comment and docstring share a line (rare) or empty line in docstring
                    # Let's do a line-by-line classification
                    dense_count = 0
                    for i, line in enumerate(raw_lines, 1):
                        if not line.strip():
                            continue # Empty
                        if i in docstring_lines:
                            continue # Docstring
                        if line.strip().startswith('#'):
                            continue # Pure comment line

                        # It's a code line!
                        dense_count += 1

                    total_dense_lines += dense_count

                except Exception:
                    pass

    return total_files, total_raw_lines, total_dense_lines, total_empty, total_comments, total_docstrings

exclude = {'node_modules', '__pycache__', 'outputs', 'data', 'venv', '.venv', 'build', 'dist', 'locks'}
paths = ['src', 'tests', 'validation']

files, raw, dense, empty, comments, docs = count_dense_python(paths, exclude)
print(f"Total Python Files: {files}")
print(f"Raw Lines (including tests/src): {raw}")
print(f"Empty Lines: {empty}")
print(f"Pure Comment Lines: {comments}")
print(f"Docstring Lines: {docs}")
print("---")
print(f"DENSE CODE LINES (Executable / Definitions): {dense}")
print(f"Compression Ratio: {dense/raw*100:.1f}% of original size")
