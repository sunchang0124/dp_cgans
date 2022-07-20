#!/bin/bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
export PATH="$HOME/.poetry/bin:$PATH"
poetry env use $(which python)
poetry install
poetry run python OWL2Vec-Star/setup.py install