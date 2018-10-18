from keras.preprocessing.text import Tokenizer
from extract_text_and_labels import *
import sys

def load_text(json_file):
	mit_confessions = load_json_data(json_file)
	data = extract_text_and_labels(mit_confessions)
	texts = [item[CONFESSION_TEXT_INDEX] for item in data]
	return texts

def create_text_embedding(text):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(text)
	return tokenizer

if __name__ == '__main__':
	if len(sys.argv) < 2: print("Usage: " + sys.argv[0] + " json_input_file")
	text = load_text(sys.argv[1])
	tokenizer = create_text_embedding(text)


