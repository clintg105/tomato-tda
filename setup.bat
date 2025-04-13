@echo off

IF NOT EXIST "venv\Scripts\activate.bat" (
    pyenv install -s 3.9.13
    pyenv local 3.9.13
    python -m venv venv
)

.\venv\Scripts\activate.bat
pip install -r requirements.txt