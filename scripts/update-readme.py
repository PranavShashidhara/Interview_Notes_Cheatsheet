#!/usr/bin/env python3
"""Update README.md with auto-generated documentation index."""

import re
import sys

def update_readme(readme_path, docs_content):
    """Replace docs content between markers in README.md."""
    try:
        with open(readme_path, 'r') as f:
            content = f.read()

        # Pattern to match the markers
        pattern = r"(<!-- DOCS_START.*?-->\n)(.*?)(\n?<!-- DOCS_END -->)"

        # Replace content between markers
        new_content = re.sub(
            pattern,
            r"\1" + docs_content.rstrip() + r"\n\3",
            content,
            flags=re.DOTALL
        )

        with open(readme_path, 'w') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error updating README.md: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: update-readme.py <docs_file>", file=sys.stderr)
        sys.exit(1)

    docs_file = sys.argv[1]

    try:
        with open(docs_file, 'r') as f:
            docs_content = f.read()
    except FileNotFoundError:
        print(f"Error: {docs_file} not found", file=sys.stderr)
        sys.exit(1)

    if update_readme("README.md", docs_content):
        sys.exit(0)
    else:
        sys.exit(1)
