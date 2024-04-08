## Create virtual environment
python -m venv main_env
## Activate the environment
source ./main_env/bin/activate
## Install requirements
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install --upgrade pip
pip install -r requirements.txt
## Deactivate the environment
deactivate 