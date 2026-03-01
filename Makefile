.PHONY: install dev-install test lint format clean run-api run-dashboard docker-build docker-up docker-down

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

lint:
	flake8 urbanai/
	mypy urbanai/

format:
	black urbanai/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

run-api:
	python -m urbanai.backend_api.main

run-dashboard:
	cd dashboard && npm run dev

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
