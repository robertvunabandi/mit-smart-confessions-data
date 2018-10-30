import data.data_util
import models.parsing.tokenizer as text_tokenizer
import models.storage.store as model_store
from keras import Sequential, Model, layers, optimizers
from models.plotting.plot import plot_regression_history, plot_prediction


DEFAULT_FILE_NAME = "all_confessions/all"
EMBEDDING_SIZE = 16
EPOCHS = 250
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.33


# Tensorflow Example:
# https://www.tensorflow.org/tutorials/keras/basic_regression


def create_regression_model(number_of_words: int, input_length: int) -> Model:
	"""
	:param number_of_words : int
	:param input_length : in
	:return Model
	"""
	model = Sequential()
	model.add(layers.Embedding(number_of_words, EMBEDDING_SIZE, input_length=input_length))
	model.add(layers.GlobalAveragePooling1D())
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dropout(0.25))
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dropout(0.1))
	model.add(layers.Dense(1, activation="linear"))
	model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])
	model.summary()
	return model


def train_regression_model() -> model:
	texts, like_labels = data.data_util.load_text_with_specific_label(
		DEFAULT_FILE_NAME,
		data.data_util.FbReaction.LIKE_INDEX
	)
	standardized_labels, avg, std = data.data_util.standardize_array(like_labels)
	sequences, num_words, index_to_word, word_to_index = text_tokenizer.get_text_items(texts)
	train_data, train_labels, test_data, test_labels, max_sequence_length = \
		text_tokenizer.split_dataset(sequences, standardized_labels, word_to_index)
	model = create_regression_model(num_words, max_sequence_length)
	history = model.fit(
		train_data,
		train_labels,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		validation_split=VALIDATION_SPLIT,
		verbose=1,
	)
	loss, accuracy = model.evaluate(test_data, test_labels)
	print("test set results: loss: %f, mean absolute error: %f" % (loss, accuracy))
	plot_regression_history(history)
	plot_prediction(model, train_data, train_labels, "train")
	plot_prediction(model, test_data, test_labels, "test")
	model_store.save_model(
		model,
		"regression",
		EMBEDDING_SIZE,
		EPOCHS,
		BATCH_SIZE,
		VALIDATION_SPLIT,
	)
	return model


if __name__ == "__main__":
	model = train_regression_model()
	# todo - make some predictions after
