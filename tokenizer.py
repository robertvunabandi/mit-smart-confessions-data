from keras.preprocessing.text import Tokenizer
from extract_text_and_labels import *
from random import shuffle
import sys

def load_data(json_file):
	mit_confessions = load_json_data(json_file)
	data = extract_text_and_labels(mit_confessions)
	shuffle(data)
	texts = [item[CONFESSION_TEXT_INDEX] for item in data]
	labels = [item[CONFESSION_REACTION_INDEX] for item in data]
	return (texts, labels)

def create_text_encoding(text):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(text)
	return tokenizer.texts_to_sequences(text)

if __name__ == '__main__':
	if len(sys.argv) < 2: print("Usage: " + sys.argv[0] + " json_input_file")
	text, labels = load_data(sys.argv[1])
	encoded_text = create_text_encoding(text)



