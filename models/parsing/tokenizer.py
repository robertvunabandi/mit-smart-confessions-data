from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from typing import Dict, List, Tuple, Any
import numpy as np
import utils


# words to filter out of the strings
FILTERS = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	'
PAD, START, UNKNOWN, UNUSED = "<PAD>", "<START>", "<UNK>", "<UNUSED>"
# todo - figure out whether we want to use this or not
# additional words to add to the tokenizer
# the Tokenizer object will ignore the words that it doesn't know
# when making predictions. adding these words can help the tokenizer
# recognize them. add each words to this list
ADDITIONAL_WORDS = []


# ******************************************
# Main Methods:
# Use these to tokenize text, split datasets
# into test and train, and to convert text
# into sequences and vice versa
# ******************************************


def get_text_items(text_array: List[str]) -> Tuple:
	"""
	:param text_array : list[str]
	:return Tokenizer
	"""
	tokenizer = Tokenizer(filters=FILTERS, split=" ")
	tokenizer.fit_on_texts(list(text_array) + ["".join(ADDITIONAL_WORDS)])
	index_to_word, word_to_index = _word_index_maps(tokenizer)
	num_words = len(word_to_index)
	sequences = _update_sequences_with_new_word_index_map(
		tokenizer.texts_to_sequences(text_array),
		tokenizer.index_word,
		word_to_index,
	)
	return sequences, num_words, index_to_word, word_to_index


def split_dataset(
	sequences: List[List[int]],
	labels: List[int] or Tuple[int, ...],
	word_to_index: Dict[str, int],
	test_percentage: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
	"""
	:param sequences : list[list[int]]
	:param labels : list[int]|tuple[int]
	:param word_to_index : dict<str, int>
	:param test_percentage : float
		of all the data, how many are used for the test set
	:return : tuple<np.array, np.array, np.array, np.array, int>
		-> training data, training labels, testing data, testing labels, max sequence length
	"""
	max_sequence_length = _find_max_sentence_length(sequences)
	split_index = len(sequences) - int(test_percentage * len(sequences))
	raw_train_data, raw_train_labels = sequences[:split_index], list(labels[:split_index])
	raw_test_data, raw_test_labels = sequences[split_index:], list(labels[split_index:])
	train_data = pad_sequences(
		raw_train_data, value=word_to_index[PAD], maxlen=max_sequence_length, padding="post"
	)
	test_data = pad_sequences(
		raw_test_data, value=word_to_index[PAD], maxlen=max_sequence_length, padding="post"
	)
	if len(np.array(labels).shape) == 1:
		train_labels = np.array([raw_train_labels]).T
		test_labels = np.array([raw_test_labels]).T
	else:
		train_labels = np.array(raw_train_labels)
		test_labels = np.array(raw_test_labels)
	return train_data, train_labels, test_data, test_labels, max_sequence_length


def convert_text_to_sequence(text: str, word_to_index: Dict[str, int]) -> list:
	"""
	:param text : str
	:param word_to_index : dict<str, int>
	:return list[int]
	"""
	words = utils.Str.extract_words_from_text_with_filters(text, FILTERS)
	return [word_to_index.get(word, word_to_index[UNKNOWN]) for word in words]


def convert_sequence_to_text(sequence: List[int], index_to_word: Dict[int, str]) -> str:
	"""
	:param sequence : list[int]
	:param index_to_word : dict<int, str>
	:return str
	"""
	return ' '.join([index_to_word.get(index, UNKNOWN) for index in sequence])


# ************************************
# Private Methods To Be Used Here Only
# ************************************

def _word_index_maps(tokenizer: Tokenizer) -> Tuple[Dict[int, str], Dict[str, int]]:
	"""
	return dictionaries that can be used to map indices
	to words and vice versa
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


def _update_sequences_with_new_word_index_map(
	raw_sequences: List[List[int]],
	raw_index_to_word: Dict[int, str],
	word_to_index: Dict[str, int]) -> List[List[int]]:
	"""
	Converts the raw sequences to a sequence with an updated word_to_index
	dictionary. To do that, we need the original index_to_word, which is
	given in raw_index_to_word.
	"""
	return [
		[word_to_index[raw_index_to_word[index]] for index in sequence]
		for sequence in raw_sequences
	]


def _find_max_sentence_length(sequences: list) -> int:
	"""
	:param sequences : list[list[int]]
	:return int
	"""
	max_length = None
	for seq in sequences:
		if max_length is None or len(seq) > max_length:
			max_length = len(seq)
	# return the next power of 2, there's no reason why do this lol.
	length_power_2 = 1
	while length_power_2 < max_length:
		length_power_2 *= 2
	return length_power_2
