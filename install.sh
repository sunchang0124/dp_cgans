#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
python -m ipykernel install --user --name venv