
# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_BASE=$(conda info --base)
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

  quality_checks:
  	## quality checks: isort, black, pylint
	$(CONDA_ACTIVATE) kitchenware_classification
	isort .
	black .
	pylint --recursive=y .

  setup:
  	## create conda env from .yaml file
	conda env create --file=kitchenware_classification.yaml
	pre-commit install

  update:
  	## update existing env from .yaml file
	conda env update --prune -f kitchenware_classification.yaml

  train_debug:
  	## train in debug mode (using only a small subset of data)
	$(CONDA_ACTIVATE) kitchenware_classification
	echo $$(which python)
	python3 train.py --debug --batch-size 16

  train:
	## train model
	$(CONDA_ACTIVATE) kitchenware_classification
	echo $$(which python)
	python3 train.py --backbone alexnet --batch-size 8

  train_nni:
	$(CONDA_ACTIVATE) kitchenware_classification
	echo $$(which python)
	#port=$$((8080 + $$(RANDOM) % 10000))
	#port=8080
	#echo $(port)
	nnictl create -c /home/frauke/kitchenware-classification/nni/config/config_local.yml -p 8080

	#nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port || nnictl create -c $configdir/config.yml -p $port
	sleep 23h 50m
	nnictl stop

  train_levante:
  	## train on levante
	sbatch jobs/start_job.sh

  prediction:
  	## load trained model and make predictions on test set
	$(CONDA_ACTIVATE) kitchenware_classification
	echo $$(which python)
	python3 prediction.py
