from flask import Flask, request
import data.data_util
from models.binary_classification import BinaryClassification

HOST = '0.0.0.0'
PORT = 5000

app = Flask(__name__)

@app.route('/')
def index():
    return 'MIT Smart Confessions API'


@app.route('/predict/classify', methods = ['POST', 'GET'])
def classify():
    text = request.args.get("text")
    bc = BinaryClassification(data.data_util.FbReaction.LIKE_INDEX)
    bc.load()
    d = bc.convert_text_to_padded_sequence(text)
    result = bc.predict(d)
    return str(result)


if __name__ == '__main__' and __package__ is None:
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT,
            threaded=False)