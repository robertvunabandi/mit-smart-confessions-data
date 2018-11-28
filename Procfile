worker: python3 -m venv virtualenv
worker: source virtualenv/bin/activate
worker: pip3 install -r requirements.txt
web: APP_HOST=$HOST APP_PORT=$PORT python3 main.py