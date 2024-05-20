
# Run setup from assignment_4 folder 

## Create virtual environment
python -m venv VA_A4_env
## Activate the environment
source ./VA_A4_env/bin/activate
## Install requirements
sudo apt-get update
sudo apt-get install libgl1
pip install opencv-python-headless
pip install --upgrade pip
pip install -r requirements.txt
## Deactivate the environment
deactivate 


