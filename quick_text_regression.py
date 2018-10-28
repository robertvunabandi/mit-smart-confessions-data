import json
import re
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential, Model, layers, optimizers
import numpy as np
import matplotlib.pyplot as plt


# https://www.tensorflow.org/tutorials/keras/basic_text_classification
# https://www.tensorflow.org/tutorials/keras/basic_regression

DEFAULT_FILE_NAME = "mit_confessions_feed_array"

"""
LOADING THE TEXT AND EXTRACTING KEY DETAILS
"""

DEFAULT_COMMENT_COUNT = 0
CONFESSION_ID_INDEX = 0
CONFESSION_TEXT_INDEX = 1
CONFESSION_REACTION_INDEX = 2
FB_REACTIONS = [fb_type.lower() for fb_type in ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]]
FB_REACTION_LIKE_INDEX = 0
FB_REACTION_LOVE_INDEX = 1
FB_REACTION_WOW_INDEX = 2
FB_REACTION_HAHA_INDEX = 3
FB_REACTION_SAD_INDEX = 4
FB_REACTION_ANGRY_INDEX = 5


def load_text_with_labels(file_name: str, label_index: int) -> tuple:
	"""
	:param file_name : str
	:param label_index : str
	:return tuple<list[str], list[int]>
	"""
	data = extract_text_and_labels(load_text(file_name))
	random.shuffle(data)
	texts, labels = zip(*[
		(row[CONFESSION_TEXT_INDEX], row[CONFESSION_REACTION_INDEX][label_index])
		for row in data
	])
	return texts, labels


def load_text(file_name: str) -> list:
	"""
	:param file_name : str
	:return list[dict<str, int|str>]
	"""
	with open("data/%s.json" % file_name, "r") as file:
		return json.load(file)


def extract_text_and_labels(feed: list) -> list:
	"""
	:param feed : list[PostObject]
	:return : list[tuple<int, str, tuple[int]>]
		-> (confession_number, text, labels)
	"""
	extracted = []
	for post_obj in feed:
		raw_text = post_obj["message"]
		matches_confession_number = re.findall("^#\d+\s", raw_text)
		if len(matches_confession_number) == 0:
			# skip, this is not a confession, it's a post
			continue
		confession_number_string = matches_confession_number[0][1:]
		text = remove_whitespaces(raw_text[len(confession_number_string) + 1:])
		labels = get_labels(post_obj)
		extracted.append((int(confession_number_string[:-1]), text, labels))
	return extracted


def remove_whitespaces(text: str) -> str:
	"""
	:param text : str
	:return str
	"""
	return text.lstrip().rstrip()


def get_labels(post_obj: dict) -> tuple:
	"""
	:param post_obj : dict<str, T>
	:return tuple<int, int, int, int, int, int, int>
		-> (Each reactions x 6, comment_count)
	"""
	comment_count = post_obj.get("comments", DEFAULT_COMMENT_COUNT)
	return tuple(
		[post_obj.get("reactions", {}).get(fb_type, 0) for fb_type in FB_REACTIONS] + [comment_count]
	)


def standardize_labels(labels: list) -> tuple:
	"""
	:param labels : list[int]
	:return tuple<list[float], float, float>
	"""
	avg = sum(labels) / len(labels)
	std = (sum([(label - avg) ** 2 for label in labels]) / len(labels)) ** 0.5
	normalized_labels = [(label - avg) / std for label in labels]
	return normalized_labels, avg, std


"""
Text Embedding
"""

PAD, START, UNKNOWN, UNUSED = "<PAD>", "<START>", "<UNK>", "<UNUSED>"
# words to filter out of the strings
FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	'


def create_text_tokenizer(text_array: list) -> Tokenizer:
	"""
	:param text_array : list[str]
	:return Tokenizer
	todo - throw in random texts so that they can be recognized
	the tokenizer will simply ignore the text that it doesn't know... and that
	is not good. the words to add can all fit in just one gibberish sentence
	containing all of them. however, what words to add though?
	"""
	tokenizer = Tokenizer(filters=FILTERS, split=" ")
	# todo - add some additional words to this string
	additional_words = ""
	tokenizer.fit_on_texts(list(text_array) + [additional_words])
	return tokenizer


def get_word_index_maps(tokenizer: Tokenizer) -> tuple:
	"""
	:param tokenizer : Tokenizer
	:return tuple<dict<int, str>, dict<str, int>>
	"""
	word_to_index = {k: (v + 3) for k, v in tokenizer.word_index.items()}
	word_to_index[START] = 0
	word_to_index[PAD] = 1
	word_to_index[UNKNOWN] = 2
	word_to_index[UNUSED] = 3
	index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])
	return index_to_word, word_to_index


def update_sequences_with_new_word_index_map(
	raw_sequences: list,
	raw_index_to_word: dict,
	word_to_index: dict) -> list:
	"""
	:param raw_sequences : list[list[int]]
	:param raw_index_to_word : dict<int, str>
	:param word_to_index : dict<str, int>
	:return list[list[int]]
	"""
	sequences = []
	for sequence in raw_sequences:
		sequences.append([word_to_index[raw_index_to_word[index]] for index in sequence])
	return sequences


def convert_text_to_sequence(text: str, word_to_index: dict) -> list:
	"""
	:param text : str
	:param word_to_index : dict<str, int>
	:return list[int]
	"""
	words = extract_words_from_text_with_filters(text, FILTERS)
	return [word_to_index.get(word, word_to_index[UNKNOWN]) for word in words]


def convert_sequence_to_text(sequence: list, index_to_word: dict) -> str:
	"""
	:param sequence : list[int]
	:param index_to_word : dict<int, str>
	:return str
	"""
	return ' '.join([index_to_word.get(index, UNKNOWN) for index in sequence])


def extract_words_from_text_with_filters(text: str, char_filters: str) -> list:
	"""
	:param text : str
	:param char_filters : str
	:return list[str]
	"""
	new_text = text.lower()
	for char_filter in iter(char_filters):
		new_text = "".join(new_text.split(char_filter))
	return new_text.split(" ")


"""
Train data
"""


def get_train_and_test_data(sequences: list, labels: list, word_to_index: dict) -> tuple:
	"""
	:param sequences : list[list[int]]
	:param labels : list[int]|tuple[int]
	:param word_to_index : dict<str, int>
	:return tuple<np.array, np.array, np.array, np.array>
	todo - will need to normalize the outputs
	that will then need to be unormalized after predicting
	"""
	max_sequence_length = find_max_sentence_length(sequences)
	length = len(sequences)
	half = length // 2 + 1
	t_quarter = length - (length // 4)
	raw_train_data, raw_train_labels = sequences[:half], list(labels[:half])
	raw_test_data, raw_test_labels = sequences[half:t_quarter], list(labels[half:t_quarter])
	# todo - not sure why, but need to exclude last 4 or tf throws an error
	raw_val_data, raw_val_labels = sequences[t_quarter:length - 4], list(labels[t_quarter:length - 4])
	# padding="post" makes the pad happen at the end instead of beginning
	train_data = pad_sequences(
		raw_train_data, value=word_to_index[PAD], maxlen=max_sequence_length, padding="post"
	)
	test_data = pad_sequences(
		raw_test_data, value=word_to_index[PAD], maxlen=max_sequence_length, padding="post"
	)
	val_data = pad_sequences(
		raw_val_data, value=word_to_index[PAD], maxlen=max_sequence_length, padding="post"
	)
	train_labels = np.array([raw_train_labels]).T
	test_labels = np.array([raw_test_labels]).T
	val_labels = np.array([raw_val_labels]).T
	return train_data, train_labels, test_data, test_labels, val_data, val_labels, max_sequence_length


def find_max_sentence_length(tokenized_sequences: list) -> int:
	"""
	:param tokenized_sequences : list[list[int]]
	:return int
	"""
	max_length = None
	for seq in tokenized_sequences:
		if max_length is None or len(seq) > max_length:
			max_length = len(seq)
	# return the next power of 2, there's no reason why do this lol.
	length_power_2 = 1
	while length_power_2 < max_length:
		length_power_2 *= 2
	return length_power_2


"""
Create the Model
"""

# it's not good to put global variable at the top, but I am keeping this
# here for scoping. this is only used in the model creation below
EMBEDDING_SIZE = 16


def create_model(number_of_words: int, input_length: int) -> Model:
	"""
	:param number_of_words : int
	:param input_length : int
	:return Model
	"""
	model = Sequential()
	model.add(layers.Embedding(number_of_words, EMBEDDING_SIZE, input_length=input_length))
	model.add(layers.GlobalAveragePooling1D())
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dropout(0.25))
	model.add(layers.Dense(64, activation="relu"))
	# model.add(layers.Dropout(0.1))
	model.add(layers.Dense(1, activation="linear"))
	# model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
	model.compile(optimizer="adam", loss="mse", metrics=["mae"])
	model.summary()
	return model


"""
Running the Methods
"""


def run_mit_confessions():
	texts, like_labels = load_text_with_labels(DEFAULT_FILE_NAME, FB_REACTION_LIKE_INDEX)
	standard_like_labels, avg, std = standardize_labels(like_labels)
	tokenizer = create_text_tokenizer(texts)
	index_to_word, word_to_index = get_word_index_maps(tokenizer)
	sequences = update_sequences_with_new_word_index_map(
		tokenizer.texts_to_sequences(texts),
		tokenizer.index_word,
		word_to_index
	)
	train_data, train_labels, test_data, test_labels, val_data, val_labels, max_sequence_length = \
		get_train_and_test_data(sequences, standard_like_labels, word_to_index)
	# for t in [train_labels, test_labels, val_labels]:
	# 	t[t <= 20] = 0
	# 	t[t > 20] = 1
	print(tokenizer.num_words)
	print(train_data.shape)
	print(train_labels.shape)
	print(test_data.shape)
	print(test_labels.shape)
	print(val_data.shape)
	print(val_labels.shape)
	model = create_model(len(tokenizer.word_index), max_sequence_length)
	history = model.fit(
		train_data,
		train_labels,
		epochs=250,
		batch_size=64,
		validation_data=(val_data, val_labels),
		verbose=1
	)
	results = model.evaluate(test_data, test_labels)
	print(results)
	plot_history(history)
	plot_prediction(model, test_data, test_labels, "testing")
	plot_prediction(model, train_data, train_labels, "training")


"""
Plotting
"""


def plot_history(history):
	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [1000$]')
	plt.plot(
		history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss'
	)
	plt.plot(
		history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss'
	)
	plt.legend()
	plt.ylim([0, 1])
	plt.show()


def plot_prediction(model, test_data, test_labels, tag):
	test_predictions = model.predict(test_data).flatten()
	print(test_predictions)
	plt.scatter(test_labels, test_predictions)
	plt.xlabel('True Values (%s)' % tag)
	plt.ylabel('Predictions (%s)' % tag)
	plt.axis('equal')
	plt.xlim(plt.xlim())
	plt.ylim(plt.ylim())
	plt.plot([-5, 20], [-5, 20])
	plt.show()


if __name__ == '__main__':
	run_mit_confessions()

"""
"""
