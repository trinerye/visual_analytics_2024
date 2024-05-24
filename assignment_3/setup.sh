# Run setup from assignment_3 folder 

## Create virtual environment
python -m venv VA_A3_env
## Activate the environment
source ./VA_A3_env/bin/activate
## Install requirements
sudo apt-get update
pip install --upgrade pip
pip install -r requirements.txt
## Deactivate the environment
deactivate 