#!/bin/sh

if [ ! -f "venv/bin/activate" ]; then
    pyenv install -s 3.9.13
    pyenv local 3.9.13
    python -m venv venv
fi

. venv/bin/activate
pip install -r requirements.txt