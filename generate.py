from tokenizer import *
# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

# set seeds for reproducability
from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate(text_list):
	corpus = clean(text_list)
	tokenizer = Tokenizer()
	input_sequences, total_words = get_sequence_of_tokens(corpus, tokenizer)
	predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, total_words)
	model = create_model(max_sequence_len, total_words)
	model.fit(predictors, label, epochs=100, verbose=2)
	model.save('mit_sc_1.h5')

	examples = [("white people", 20), ("gpa", 40), ("friends", 20), ("immigrants", 30)]
	for seed_text, next_words in examples:
		print(generate_text(tokenizer, seed_text, next_words, model, max_sequence_len))



def clean(text_list) -> list:
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
    model.add(Embedding(total_words, 10, input_length=input_len))

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

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# convenience function for debugging
def load_mit_confessions():
	return load_data("mit_confessions_feed_array")


if __name__ == '__main__':
	if len(sys.argv) < 2: print("Usage: " + sys.argv[0] + " json_input_file")
	text_list, _ = load_data(sys.argv[1]) # ignore labels for generator
	generate(text_list)







