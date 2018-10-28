# MIT Smart Confessions DATA

## Installation

To create the virtualenv dev location (don't do this 
again because it's done in this repo):

	virtualenv facebookenv

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