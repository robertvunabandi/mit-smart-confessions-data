# MIT Smart Confessions Api

## Installation

Make sure you are using Python 3.6.x (3.6.7 preferably). Tensorflow will not run correctly with python 3.7.0. See [this link](https://www.python.org/downloads/release/python-367/) in which you can download python 3.6.7 (scroll to the bottom of the page). 

To create the virtualenv dev location (don't do this 
again because it's done in this repo) (more instructions [here](https://www.caseylabs.com/how-to-create-a-python-3-6-virtual-environment-on-ubuntu-16-04/)):

	python3.6 -m venv virtualenv facebookenv # ensures a python 3.6 virtualenv

To activate, run:

	source facebookenv/bin/activate

To install the tool to use FB API (don't do this 
again because it's done in this repo):
	
	pip3 install -r requirements.txt
	pip3 install facebook-sdk

To deactivate, run:
	
	deactivate

See https://facebook-sdk.readthedocs.io/en/latest/install.html for details.

To install `tensorflow 1.11.0` with a link (needed for `python 7.x`), run:

    pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.11.0-py3-none-any.whl