worker: python3 -m venv virtualenv
worker: source virtualenv/bin/activate
worker: pip3 install -r requirements.txt
web: FLASK_APP=main.py flask run --host=$HOST --port=$PORT