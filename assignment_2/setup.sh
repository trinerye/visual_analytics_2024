# Run setup from assignment_2 folder 

## Create virtual environment
python -m venv VA_A2_env
## Activate the environment
source ./VA_A2_env/bin/activate
## Install requirements
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install --upgrade pip
pip install -r requirements.txt
## Deactivate the environment
deactivate 