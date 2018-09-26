# Makefile to build, test and deploy automl_toolbox
#

REQUIREMENTS_FILE = requirements.txt

#########
# Build
#########

.PHONY: requirements
requirements:
	pipenv lock --requirements > $(REQUIREMENTS_FILE)

.PHONY: build
build: mybuild
mybuild:
	make requirements
	python setup.py develop
	rm $(REQUIREMENTS_FILE)


#########
# Test
#########

lint:
	flake8
