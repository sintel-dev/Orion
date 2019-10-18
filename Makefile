.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 orion tests
	isort -c --recursive orion tests

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find orion -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive orion
	isort --apply --atomic --recursive orion

	find tests -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive tests
	isort --apply --atomic --recursive tests

.PHONY: test
test: ## run tests quickly with the default Python
	python -m pytest tests --cov=orion

.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source orion -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc --module-first --separate -T -o docs/api/ orion
	$(MAKE) -C docs html
	grep -Rl 'href="[^"]*.md"' docs/_build/ | xargs -n1 sed -i 's/href="\([^"]*\).md"/href="\L\1.html"/g'
	grep -Rl 'docs/images' docs/_build/ | xargs -n1 sed -i 's:docs/images/:_static/:g'

.PHONY: view-docs
view-docs: docs ## view docs in browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: view-docs ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' .

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: test-publish
test-publish: dist ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish: dist ## package and upload a release
	twine upload dist/*

.PHONY: bumpversion-release
bumpversion-release: ## Merge master to stable and bumpversion release
	git checkout stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to master and bumpversion patch
	git checkout master
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CHANGELOG_LINES := $(shell git diff HEAD..stable HISTORY.md 2>/dev/null | wc -l)

.PHONY: check-release
check-release: ## Check if the release can be made
ifneq ($(CURRENT_BRANCH),master)
	$(error Please make the release from master branch\n)
endif
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: release
release: check-release bumpversion-release publish bumpversion-patch

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed



.PHONY: docker-jupyter-clean
docker-jupyter-clean: ## Remove the orion-jupyter docker image
	docker rmi -f orion-jupyter

.PHONY: docker-jupyter-build
docker-jupyter-build: ## Build the orion-jupyter docker image using repo2docker
	docker build -f docker/orion-jupyter.Dockerfile -t orion-jupyter .

.PHONY: docker-jupyter-save
docker-jupyter-save: docker-jupyter-build  ## Build the orion-jupyter image and save it as orion-jupyter.tar
	docker save --output orion-jupyter.tar orion-jupyter

.PHONY: docker-jupyter-load
docker-jupyter-load: ## Load the orion-jupyter image from orion-jupyter.tar
	docker load --input orion-jupyter.tar

.PHONY: docker-jupyter-run
docker-jupyter-run: ## Run the orion-jupyter image in editable mode
	docker run --rm -v $(shell pwd):/app -ti -p8888:8888 --name orion-jupyter orion-jupyter

.PHONY: docker-jupyter-start
docker-jupyter-start: ## Start the orion-jupyter image as a daemon
	docker run --rm -d -v $(shell pwd):/app -ti -p8888:8888 --name orion-jupyter orion-jupyter

.PHONY: docker-jupyter-stop
docker-jupyter-stop: ## Stop the orion-jupyter daemon
	docker stop orion-jupyter

.PHONY: docker-distribute
docker-jupyter-package: docker-jupyter-save ## Build the docker-jupyter image and package it for distribution
	docker/package.sh

