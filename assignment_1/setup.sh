# Run setup from assignment 1 folder 

## create virtual env
python -m venv A1_env
## activate env
source ./A1_env/bin/activate
## install requirements
pip install --upgrade pip
pip freeze > requirements.txt
pip install -r requirements.txt
## deactivate env
deactivate 