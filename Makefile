.PHONY: default clean doctest lint pre-commit typecheck

default:
	echo "Hello, World!"

clean:
	find . -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist

docs:
	uv run pdoc -d google --math --no-include-undocumented tabm.py

doctest:
	uv run xdoctest tabm.py
	uv run test_code_blocks.py README.md

lint:
	uv run typos tabm.py
	uv run typos example.ipynb
	uv run typos README.md
	uv run ruff check tabm.py
	uv run ruff format --check tabm.py

# The order is important.
pre-commit: clean lint doctest typecheck

typecheck:
	uv run mypy tabm.py
