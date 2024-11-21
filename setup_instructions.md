## for setting up conda environment:
 -> conda env create -f environment.yml
## for install required python packes
 -> Inside docker container
	* Dcokerfile ensures installation of all dependencies listed in requirements.txt
 -> In personal machine
	* pip install -r requirements.txt
## for installing the atommks package in dev mode
	* pip install -e .
## for running jupyter notebook
	* jupyter-notebook --ip 0.0.0.0 --port 8000 --allow-root --no-browser

