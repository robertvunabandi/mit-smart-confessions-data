import data.data_util
import models.parsing.tokenizer as text_tokenizer
import models.storage.store as model_store
import numpy as np
import keras.utils
from keras import Sequential, Model, layers
from keras.preprocessing.sequence import pad_sequences
from typing import Dict, List, Tuple


# todo set the seeds for reproducibility


DEFAULT_FILE_NAME = "all_confessions/all"
EMBEDDING_SIZE = 16
EPOCHS = 1
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
EXAMPLES = [("white people", 20), ("gpa", 40), ("friends", 20), ("immigrants", 30)]
LSTM_HIDDEN_NEURONS = 100


def create_lstm_model(number_of_words: int, input_length: int) -> Model:
	"""
	:param number_of_words : int
	:param input_length : in
	:return Model
	"""
	model = Sequential()
	model.add(layers.Embedding(number_of_words, EMBEDDING_SIZE, input_length=input_length - 1))
	model.add(layers.LSTM(LSTM_HIDDEN_NEURONS))
	model.add(layers.Dense(number_of_words, activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer="adam")
	return model


def train_lstm_model():
	"""
	Trains and returns an LSTM model
	this is extremely slow especially now that we're using 4000+ examples
	"""
	texts, _ = data.data_util.load_text_with_specific_label(
		DEFAULT_FILE_NAME,
		data.data_util.FbReaction.LIKE_INDEX
	)
	sequences, num_words, index_to_word, word_to_index = text_tokenizer.get_text_items(texts)
	predictors, labels, max_sequence_len = generate_padded_sequences(sequences, num_words, word_to_index)
	model = create_lstm_model(num_words, max_sequence_len)
	# todo - can we use batch size and validation split here?
	history = model.fit(
		predictors,
		labels,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		verbose=1,
	)
	model_store.save_model(
		model,
		"lstm_hidden_neurons_%d" % LSTM_HIDDEN_NEURONS,
		EMBEDDING_SIZE,
		EPOCHS,
		BATCH_SIZE,
		VALIDATION_SPLIT,
	)
	return model


def generate_padded_sequences(input_sequences, total_words, word_to_index):
	"""
	generates predictors, which are padded sequences up to the
	maximum sequence length calculated below. it also produces labels,
	which are the next word for the given sequence. finally, it also
	returns the maximum sequence length.

	:param input_sequences : list[list[int]]
	:param total_words : int
	:param word_to_index : dict<str, int>
	:return tuple<predictors, labels, max_sequence_length>
		-> predictors : np.ndarray
		-> labels : np.ndarray
		-> max_sequence_length : int
	"""
	max_sequence_len = max([len(sequence) for sequence in input_sequences])
	sequences_with_ends = [
		sequence + ([word_to_index[text_tokenizer.END]] if len(sequence) < max_sequence_len else [])
		for sequence in input_sequences
	]
	input_sequences = np.array(pad_sequences(
		sequences_with_ends,
		value=word_to_index[text_tokenizer.PAD],
		maxlen=max_sequence_len,
		padding="post",
	))
	# todo - we need to create a predictor and label for index up until the padding sequence
	# what we currently have is BAD because the labels at the end
	# are always the same (they are whatever the padded value was),
	# i.e. {text_tokenizer.PAD}. Instead, each sub-sequence needs to
	# have a label that is the next word up until the last word that
	# is a pad value
	predictors, raw_labels = input_sequences[:, :-1], input_sequences[:, -1]
	labels = keras.utils.to_categorical(raw_labels, num_classes=total_words)
	return predictors, labels, max_sequence_len


def load_model_and_predict(model: Model = None) -> None:
	"""
	Load the appropriate model and use it to make predictions
	"""
	texts, _ = data.data_util.load_text_with_specific_label(
		DEFAULT_FILE_NAME,
		data.data_util.FbReaction.LIKE_INDEX
	)
	sequences, num_words, index_to_word, word_to_index = text_tokenizer.get_text_items(texts)
	_, _, max_sequence_len = generate_padded_sequences(sequences, num_words, word_to_index)
	if model is None:
		model = create_lstm_model(num_words, max_sequence_len)
	model_name = model_store.get_model_title(
		"lstm_hidden_neurons_%d" % LSTM_HIDDEN_NEURONS,
		EMBEDDING_SIZE,
		EPOCHS,
		BATCH_SIZE,
		VALIDATION_SPLIT,
	)
	model.load_weights(model_name)
	output = predict_from_example_list(model, word_to_index, max_sequence_len, EXAMPLES)
	print(output)


def predict_from_example_list(
	model: Model,
	word_to_index: Dict[str, int],
	max_sequence_len: int,
	examples: List[Tuple[str, int]]) -> List[Tuple[str, int, str]]:
	"""

	:param model : Model
	:param word_to_index : dict<str, int>
	:param max_sequence_len : int
	:param examples : list[tuple<str, int>]
	:return list[tuple<seed_text, int, prediction>]
		-> seed_text : str
		-> prediction : str
	"""
	predictions = []
	for seed_text, total_word_additions in examples:
		example_prediction = generate_text(
			model,
			word_to_index,
			seed_text,
			total_word_additions,
			max_sequence_len,
		)
		predictions.append((seed_text, total_word_additions, example_prediction))
	return predictions


def generate_text(model, word_to_index, seed_text, total_word_additions: int, max_sequence_len):
	"""
	:param model : Model
	:param word_to_index : dict<str, int>
	:param seed_text : str
	:param total_word_additions : int
	:param max_sequence_len : int
	:return str
	"""
	output_sequence = text_tokenizer.convert_text_to_sequence(seed_text, word_to_index)
	initial_length = len(output_sequence)
	padded_sequence = pad_sequences([output_sequence], maxlen=max_sequence_len - 1, padding="post")
	for index in range(initial_length, total_word_additions + initial_length):
		predicted_index = model.predict_classes(padded_sequence, verbose=0)[0]
		output_sequence.append(predicted_index)
		padded_sequence[0, index] = predicted_index
	index_to_word = dict([(index, word) for word, index in word_to_index.items()])
	return text_tokenizer.convert_sequence_to_text(output_sequence, index_to_word)


is_training = True
if __name__ == "__main__":
	# we want to give the computer some times after training to
	# store the model. then, we can load it. so, turn is_training
	# to true to train instead of make predictions.
	model_ = None
	if is_training:
		model_ = train_lstm_model()
	# predictions are bad right now because all labels are
	# the same, so it keeps predicting the same label for each
	load_model_and_predict(model_)
