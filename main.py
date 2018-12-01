#!/usr/bin/python3
"""
run with:
FLASK_APP=main.py flask run

or, to specify the host and port, run with:
FLASK_APP=main.py flask run --host={some-host} --port={some-port}
"""
import os
from flask import Flask, request
import tensorflow as tf

import utils
import constants
from models.bucket_classification import BucketClassification
from models.lstm_generator import LSTMGenerator


###########
# GLOBALS #
###########

# the following are not really constants, but they are
# global variables. whenever we load a model, it will
# cache the model into these variables to use them later
# on.
CLASSIFIER_MODELS = {}
LSTM_MODELS = {}
DEFAULT_GENERATE_LENGTH = 20
# using the tf graph solves the problem that we were having
# about the lstm working or not working (vice versa with the
# classifier). I still am not sure why this fixes it, but it
# seems like a problem due to asynchronous-ness or the way
# threading works when using flask. In addition, it could be
# a problem with the python version. Not sure but this fixes
# it with calling "with tf_graph.as_default():" then predicting
# within that scope.
TF_GRAPH = {}
HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", 5000)

#########################
# Route for Predictions #
#########################

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
    bc = BucketClassification(index, should_depolarize=False)
    bc.load()
    CLASSIFIER_MODELS[index] = bc
    return CLASSIFIER_MODELS[index]


@app.route("/predict", methods=["GET"])
def classify():
    text = request.args.get("text")
    result = {}
    # for each index, make a prediction
    for index, reaction in enumerate(constants.FB_REACTIONS):
        bc = get_classifier_model(index)
        d = bc.convert_text_to_padded_sequence(text)
        with TF_GRAPH[0].as_default():
            prediction = bc.predict(d).tolist()[0]
            result[reaction.lower()] = []
            for zipped in zip(bc.bucket_ranges, prediction):
                ranges, prob = zipped
                min_v, max_v = ranges
                result[reaction.lower()].append([min_v, max_v, prob])
    return utils.make_string_json_valid(str(result))


def get_lstm_model():
    if 0 in LSTM_MODELS:
        return LSTM_MODELS[0]
    LSTM_MODELS[0] = LSTMGenerator(popularity_threshold=40)
    LSTM_MODELS[0].load()
    return LSTM_MODELS[0]


@app.route("/generate", methods=["GET"])
def generate():
    seed = request.args.get("seed")
    try:
        length = int(request.args.get("length", DEFAULT_GENERATE_LENGTH))
    except Exception:
        length = DEFAULT_GENERATE_LENGTH
    lstm_model = get_lstm_model()
    with TF_GRAPH[0].as_default():
        return lstm_model.generate(seed, length)


##################
# Error Handling #
##################

@app.errorhandler(404)
def error_not_found(error):
    return "Not found: %s" % str(error), 404


@app.errorhandler(500)
def error_not_found(error):
    return "Internal Server Error - %s" % str(error), 500


####################
# Model Preloading #
####################

def preload_models():
    get_lstm_model()
    for index in range(len(constants.FB_REACTIONS)):
        get_classifier_model(index)
    TF_GRAPH[0] = tf.get_default_graph()


# preload the models
preload_models()

if __name__ == "__main__":
    app.run(
            host=HOST,
            debug=False,  # automatic reloading enabled
            threaded=True,
            port=PORT,
    )
