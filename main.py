"""
run with:
FLASK_APP=main.py flask run

or, to specify the host and port, run with:
FLASK_APP=main.py flask run --host={some-host} --port={some-port}
"""
import os
from flask import Flask, request

import utils
import constants

from models.bucket_classification import BucketClassification
import legacy_lstm_model


###########
# GLOBALS #
###########

# the following are not really constants, but they are
# global variables. whenever we load a model, it will
# cache the model into these variables to use them later
# on.
CLASSIFIER_MODELS = {}
LSTM_MODEL = None
HOST = os.getenv("HOST", "0.0.0.0.")
PORT = os.getenv("HOST", 5000)

########
# CODE #
########

app = Flask(__name__)


@app.route("/")
def home():
    return "MIT Smart Confessions API"


def get_classifier_model(index: int) -> BucketClassification:
    """
    loads the classifier model based on the label index. labels
    are for confession reactions. see constant.py to see which
    index refer to which reaction.
    """
    if index in CLASSIFIER_MODELS:
        return CLASSIFIER_MODELS[index]
    bc = BucketClassification(index, should_depolarize=True)
    bc.load()
    CLASSIFIER_MODELS[index] = bc
    return CLASSIFIER_MODELS[index]


@app.route("/predict", methods=["GET"])
def classify():
    text = request.args.get("text")
    # if len(text.split(" ")) > 0:
    #     text = '"%s"' % text
    result = {}
    # for each index, make a prediction
    for index, reaction in enumerate(constants.FB_REACTIONS):
        bc = get_classifier_model(index)
        d = bc.convert_text_to_padded_sequence(text)
        prediction = bc.predict(d).tolist()[0]
        result[reaction.lower()] = []
        for zipped in zip(bc.bucket_ranges, prediction):
            ranges, prob = zipped
            min_v, max_v = ranges
            result[reaction.lower()].append([min_v, max_v, prob])
    return utils.make_string_json_valid(str(result))


def get_lstm_model():
    if LSTM_MODEL is None:
        # todo - set LSTM_MODEL to load the model
        pass
    return LSTM_MODEL


@app.route("/generate", methods=["GET"])
def generate():
    # todo - should store the model inside LSTM_MODELS
    seed = request.args.get("seed")
    length = int(request.args.get("length"))  # todo: actually check whether this is an integer
    output = legacy_lstm_model.generate_from_stored_model(seed, length)
    return output


# todo - hey Jurgen, could you add some comments about the __package__ is None check?
# todo - will need to not specify the host and port so that it runs in server
if __name__ == "__main__" and __package__ is None:
    app.run(
            host=HOST,
            debug=True,  # automatic reloading enabled
            threaded=False,
            port=PORT,
    )
