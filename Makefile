.PHONY: lint check
lint:
	python -m black outliers
check:
	python -m black outliers --check
	python -m mypy outliers

.PHONY: test
test:
	python -m pytest outliers

.PHONY: commit
commit: check test

.PHONY: setup
setup:
	git config --local core.hooksPath ./hooks/
	chmod +x ./hooks/pre-commit
	python -m venv venv
	chmod +x ./venv/bin/activate
	. venv/bin/activate; pip install -r requirements.txt