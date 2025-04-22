.PHONY: install install-dev lint

install:
	@pip install -e . 

install-dev:
	@pip install -e ".[dev]"

test:
	@pytest --verbose

lint:
	@echo "Running Ruff and Isort..."
	@isort .
	@ruff format .
	@ruff check . --fix

clean:
	@rm -rf build
	@find . -name *.so | xargs -I {} rm -f {}
	@find . -name __pycache__ | xargs -I {} rm -rf {}
	@find . -name *.pyd | xargs -I {} rm -f {}
	@find . -name *.egg* | xargs -I {} rm -rf {}