# MIT Smart Confessions API

[MIT confessions](https://www.facebook.com/beaverconfessions) is a Facebook page where MIT students posts anonymously and get reactions from other MIT students. These posts are called "confessions". 

MIT Smart Confessions is a platform that uses machine learning to predict the number of reactions one would get from their confession and to generate confessions in such a way that it maximizes the number of reactions that confession would get.

The website for this application can be found at [mit-smart-confessions.herokuapp.com](https://mit-smart-confessions.herokuapp.com), and the Github repository for the website can be found on [github.com/robertvunabandi/mit-smart-confessions-website](https://github.com/robertvunabandi/mit-smart-confessions-website).

[![MIT Smart Confession Logo](assets/msc-logo.png)](https://mit-smart-confessions.herokuapp.com)

## Installation

### Python Version: 3.6.x

Make sure you are using Python 3.6.x (3.6.7 preferably). Tensorflow will not run correctly with python 3.7.0 because python 3.7.0 has some additional reserved keywords that tensorflow used. See [this link](https://www.python.org/downloads/release/python-367/) in which you can download python 3.6.7 (scroll to the bottom of the page). 

### Python Virtual Environment

When running, **use a python virtual environment**. To create the virtual environment (more instructions [here](https://www.caseylabs.com/how-to-create-a-python-3-6-virtual-environment-on-ubuntu-16-04/)):

```bash
python3.6 -m venv {virtualenv-name} # ensures a python 3.6.x virtualenv
```

To activate, run:

```bash
source {virtualenv-name}/bin/activate
```
	
For instance, if you name your virtual env `myvenv`, then you would run:
    
```bash    
# sets up the virual env
python3.6 -m venv myvenv
# activates the virtual env
source myvenv/bin/activate
```

To deactivate the virtualenv, run:
	
```bash
deactivate
```

You would want to deactivate the environment so that your terminal instance goes back to using the normal, system-wide installed packages.

### Package Installations

After activating your virtualenv, run the following to install all the packages used in this application (**make sure you have activated the virtualenv before running this**):

	
```bash
pip3 install -r requirements.txt
```

That will install all the packages listed in the file requirements.txt.

### Adding New Packages

If you install a new package by running `pip3 install {package-name}`, make sure to update the `requirements.txt` file. To do so, just run (**make sure you have activated the virtualenv before running this and when installing a new package**): 

```bash
pip3 freeze > requirements.txt
``` 

* `pip3 freeze` lists all the packages currently installed in the environment
* `> requirements.txt` dumps that list into the `requirements.txt` file. 

## Running `main.py`

Use the following command to run the main server file:

```bash
FLASK_APP=main.py flask run
```

In addition, if you want to specify a specific host or port, you can run:

```bash
FLASK_APP=main.py flask run --host={host} --port={port}
```

## Running Other Files

It is recommended that you use [JetBrains' PyCharm IDE](https://www.jetbrains.com/pycharm/). If you prefer using your terminal, make sure that you set `PYTHONPATH` to be the root directory (i.e. where the `main.py` is located).

For instance, if running the `bucket_classification.py` model, you should run:

```bash
PYTHONPATH=.. python3 bucket_classification.py
```

Note that we used `..` (2 dots in total) because the root directory is one parent directory above the `/models` directory (in which the `bucket_classification.py` resides). In case you are two levels deep, then you wanna do `../..`. For more levels, follow the patttern accordingly. 

## Data Collection

We collected data using the Facebook API. This is not needed anymore, but if curious, we have some details below.

The `/data/scripts` should have the code used to collect data. 

We used the `facebook-sdk` package to install data. See this [link](https://facebook-sdk.readthedocs.io/en/latest/install.html) for details on how to use the facebook package to collect data.

In order to keep this app up to date, however, we will need to collect more data in the future. 