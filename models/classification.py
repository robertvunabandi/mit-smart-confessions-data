import data.data_util
import models.parsing.tokenizer as text_tokenizer
import models.storage.store as model_store
from keras import Sequential, Model, layers, optimizers
from typing import List
from models.plotting.plot import plot_classification_history, plot_prediction


DEFAULT_FILE_NAME = "all_confessions/all"
EMBEDDING_SIZE = 16
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.33


# Tensorflow Example:
# https://www.tensorflow.org/tutorials/keras/basic_text_classification


def create_classifier_model(number_of_words: int, input_length: int, output_neurons: int = 1) -> Model:
	"""
	:param number_of_words : int
	:param input_length : int
	:param output_neurons : int
		-> the number of output neurons. this must match the labels.
	:return Model
	"""
	model = Sequential()
	model.add(layers.Embedding(number_of_words, EMBEDDING_SIZE, input_length=input_length))
	model.add(layers.GlobalAveragePooling1D())
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dropout(0.25))
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dropout(0.1))
	activation = "sigmoid" if output_neurons == 1 else "softmax"
	model.add(layers.Dense(output_neurons, activation=activation))
	# todo - include other metrics such as auc, roc in the future
	# see https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
	model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
	model.summary()
	return model


def run_binary_classifier_model(index: int) -> Model:
	"""
	Trains and returns a binary classification model
	:param index : int
		-> index for data label, see data.data_util.FbReaction
	"""
	texts, like_labels = data.data_util.load_text_with_specific_label(
		DEFAULT_FILE_NAME,
		index
	)
	binary_labels = create_binary_labels_for_classification(like_labels, 20)
	sequences, num_words, index_to_word, word_to_index = text_tokenizer.get_text_items(texts)
	train_data, train_labels, test_data, test_labels, max_sequence_length = \
		text_tokenizer.split_dataset(sequences, binary_labels, word_to_index)
	model = create_classifier_model(num_words, max_sequence_length)
	history = model.fit(
		train_data,
		train_labels,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		verbose=1,
	)
	loss, accuracy = model.evaluate(test_data, test_labels)
	print("test set results: loss: %f, accuracy: %f" % (loss, accuracy))
	plot_classification_history(history)
	plot_prediction(model, train_data, train_labels, "train")
	plot_prediction(model, test_data, test_labels, "test")
	model_store.save_model(
		model,
		"binary_classification_index_%d" % index,
		EMBEDDING_SIZE,
		EPOCHS,
		BATCH_SIZE,
		VALIDATION_SPLIT,
	)
	return model


def create_binary_labels_for_classification(labels: List[int], cut_off: int = 20) -> List[int]:
	"""
	label everything below {cut_off} as 0 and anything above as 1.
	:param labels : list[int]
	:param cut_off : int
	:return list[int]
	"""
	return [1 if label > cut_off else 0 for label in labels]


if __name__ == "__main__":
	bin_model = run_binary_classifier_model(data.data_util.FbReaction.LIKE_INDEX)
