import json
# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras.utils as ku
from keras.models import load_model
import re
import sys

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed

EPOCHS = 2

MODEL_FILE = 'models/storage/legacy_lstm_100_400.h5'
set_random_seed(2)
seed(1)

import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_from_stored_model(seed_text, next_words):
    text_list = load_mit_confessions()
    model = load_model(MODEL_FILE)
    s = model.summary()
    print(s)
    return generate(text_list, model, seed_text, next_words)


def generate(text_list, model=None, seed_text=None, next_words=None):
    corpus = clean(text_list)
    tokenizer = Tokenizer()
    input_sequences, total_words = get_sequence_of_tokens(corpus, tokenizer)
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, total_words)
    # only fit if it does not exist
    if model is None:
        model = create_model(max_sequence_len, total_words)
        model.fit(predictors, label, epochs=EPOCHS, verbose=1)
        model.save(MODEL_FILE)

    if seed_text is None or next_words is None:
        examples = [("white people", 20), ("gpa", 40), ("friends", 20), ("immigrants", 30)]
        for seed_text, next_words in examples:
            print(generate_text(tokenizer, seed_text, next_words, model, max_sequence_len))
    else:
        return(generate_text(tokenizer, seed_text, next_words, model, max_sequence_len))



def clean(text_list):
    """
    series of steps to clean data for LSTM

    :param text: list[string]
    :return : list[string]
    """

    def clean_instance(text):
        text = "".join(v for v in text if v not in string.punctuation).lower()
        txt = text.encode("utf8").decode("ascii",'ignore')
        return text

    corpus = [clean_instance(x) for x in text_list]
    return corpus


def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    print(predictors)
    print(label)
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


def get_sequence_of_tokens(corpus, tokenizer):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 16, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def generate_text(tokenizer, seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        print(token_list)
        print(predicted)

        #output_word = tokenizer.word_index[predicted]
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()



FB_REACTIONS = [fb_type.lower() for fb_type in ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]]
CONFESSION_ID_INDEX = 0
CONFESSION_TEXT_INDEX = 1
CONFESSION_REACTION_INDEX = 2

def extract_text_and_labels(feed: list) -> list:
        """
        this returns, from each of the PostObject in the feed, a tuple
        such that the first item is the confession number, the second is
        the text, and the third are labels.

        :param feed : list[PostObject]
        :return : list[tuple<int, str, tuple[int]>]
        """
        extracted = []
        for post_obj in feed:
                text = post_obj["message"]
                matches_confession_number = re.findall("^#\d+\s", text)
                if len(matches_confession_number) == 0:
                        # skip, this is not a confession
                        continue
                confession_number_string = matches_confession_number[0][1:]
                new_text = text[len(confession_number_string) + 1:]
                comment_count = post_obj.get("comments", 0)
                labels = tuple([post_obj.get("reactions", {}).get(fb_type, 0) for fb_type in FB_REACTIONS] + [comment_count])
                extracted.append((int(confession_number_string[:-1]), new_text, labels))
        return extracted

def load_data(json_file):

    def load_json_data(name: str) -> list:
            with open("data/%s.json" % name, "r") as file:
                    return json.load(file)



    mit_confessions = load_json_data(json_file)
    data = extract_text_and_labels(mit_confessions)
    texts = [item[CONFESSION_TEXT_INDEX] for item in data]
    labels = [item[CONFESSION_REACTION_INDEX] for item in data]
    return (texts, labels)

def create_text_encoding(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer.texts_to_sequences(text)


# convenience function for debugging
def load_mit_confessions(cutoff_length=400):
    m, _ = load_data("mit_confessions_feed_array")
    return [s for s in m if len(s) < cutoff_length]




if __name__ == '__main__':
    #generate(text_list)
    #if len(sys.argv) < 3:
    #    print("Usage: " + sys.argv[0] + " seed_text num_words")
    #    exit()
    #seed_text = sys.argv[1]
    #num_words = int(sys.argv[2])

    seed_text, num_words = "Anime girls are better than real girls who is with us anime lovers", 20

    output = generate_from_stored_model(seed_text, num_words)
    #output = generate(load_mit_confessions())
    print(output)