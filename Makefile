.PHONY: docs-list docs-list-check help

help:
	@echo "Available commands:"
	@echo "  make docs-list         - Generate/update the documentation index in README.md"
	@echo "  make docs-list-check   - Check if markdown files are staged"

docs-list:
	@echo "Updating documentation index in README.md..."
	@bash -c 'TEMP_DOCS=$$(mktemp); bash scripts/generate-docs-index.sh > $$TEMP_DOCS && python3 scripts/update-readme.py $$TEMP_DOCS && rm -f $$TEMP_DOCS'
	@echo "✓ Documentation index updated in README.md"

docs-list-check:
	@if git diff --name-only --cached | grep -E "\.md$$" | grep -v "README.md" > /dev/null; then \
		echo "⚠ Markdown files staged. Run 'make docs-list' to update README.md"; \
		exit 1; \
	else \
		echo "✓ No markdown changes detected"; \
	fi
