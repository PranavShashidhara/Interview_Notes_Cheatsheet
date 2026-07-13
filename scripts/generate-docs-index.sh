#!/bin/bash
# Generate documentation list from all markdown files in the repository

echo "### Root Level Docs"
echo ""

# Find and list root level markdown files (excluding README.md)
find . -maxdepth 1 -name "*.md" -not -name "README.md" -not -name "DOCS_INDEX.md" | sort | while read file; do
    title=$(basename "$file" .md)
    echo "- [$title]($file)"
done

echo ""
echo "### Advanced Topics"
echo ""

# Find and list docs in subdirectories
if [ -d "bonus_advanced_topics" ]; then
    find bonus_advanced_topics -maxdepth 1 -name "*.md" -not -name "DOCS_INDEX.md" | sort | while read file; do
        title=$(basename "$file" .md)
        echo "- [$title]($file)"
    done
fi
