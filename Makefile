.PHONY: install test quick full site
install:
	pip install -e qcore -e qengine -e ".[dev,v3]"
test:
	pytest -q
quick:
	python -m trend.cli quick
full:
	python -m trend.cli full
site:
	python scripts/v3/export_site_data.py
	python -m http.server -d docs 8000
