import keras
import numpy as np
import data.data_util
import models.parsing.tokenizer
import models.storage.store
# type annotation module
from typing import List, Tuple, Union, Any


class BaseModel:
	# keys for hyperparameters
	KEY_TEST_SPLIT = "test_split"
	KEY_EPOCHS = "epochs"
	KEY_BATCH_SIZE = "batch_size"
	KEY_VALIDATION_SPLIT = "validation_split"
	KEY_EMBEDDING_SIZE = "embedding_size"
	KEY_MAX_CONFESSION_LENGTH = "max_confession_length"
	DEFAULT_HYPERPARAMETER_VALUE_MAP = {
		"test_split": 0.20,
		"epochs": 100,
		"batch_size": 64,
		"validation_split": 0.33,
		"embedding_size": 16,
		"max_confession_length": 400,
	}

	def __init__(self, model_type_name: str):
		"""
		initializes the model.
		:param model_type_name : str
			This can be "lstm", "regression", "classification", or
			anything of the sort. This model is used to to store
			the model after training.
		"""
		self.model_type_name = model_type_name
		self.model = None
		self.train_data, self.train_labels, self.test_data, self.test_labels = None, None, None, None
		# class attributes to be used at training
		self.sequences, self.labels, self.max_sequence_length = None, None, None
		self.num_words, self.word_to_index, self.index_to_word = None, None, None
		# setting up parameters to be used later when testing/using the model
		self.hyperparams = {}
		self.last_train_history = None
		# setup the model
		self.setup_base_data()

	def get_hyperparam(self, key: str) -> Any:
		"""
		gets a hyperparameter, make sure to use one of the static
		"KEY_" attributes of the BaseModel class
		"""
		BaseModel.assert_valid_key(key)
		return self.hyperparams.get(key, BaseModel.DEFAULT_HYPERPARAMETER_VALUE_MAP.get(key, None))

	def set_hyperparam(self, key: str, value: Any) -> None:
		"""
		sets hyperparameter, make sure to use one of the static "KEY_"
		attributes of the BaseModel class
		"""
		BaseModel.assert_valid_key(key)
		self.hyperparams[key] = value

	@staticmethod
	def assert_valid_key(hyperparameter_key: str) -> None:
		"""
		throws if the hyperparameter key is not amongst the
		BaseModel's hyperparameters
		"""
		key_attributes = {attr for attr in dir(BaseModel) if "KEY_" in attr}
		key_values = {getattr(BaseModel, key) for key in key_attributes}
		assert hyperparameter_key in key_values, "the key given is not a valid key attribute"

	def setup_base_data(self) -> None:
		"""
		sets up attributes to use in the class, such as
		num_words and max_sequence_length
		"""
		max_confession_length = self.get_hyperparam(BaseModel.KEY_MAX_CONFESSION_LENGTH)
		texts, labels = self.load_base_data(max_confession_length)
		sequences, num_words, index_to_word, word_to_index = models.parsing.tokenizer.get_text_items(texts)
		max_sequence_length = models.parsing.tokenizer.find_max_sentence_length(sequences)
		# update these attributes for this class instance
		self.sequences, self.max_sequence_length, self.labels = sequences, max_sequence_length, labels
		self.num_words, self.word_to_index, self.index_to_word = num_words, word_to_index, index_to_word

	def update_train_and_test_data(self):
		self.train_data, self.train_labels, self.test_data, self.test_labels = self.get_train_and_test_data()

	def get_train_and_test_data(self):
		"""
		gets the test and train data for this model
		"""
		# shuffle the data
		np_data, np_labels = self.parse_base_data(self.sequences, self.labels)
		split = int(np_data.shape[0] - (self.get_hyperparam(BaseModel.KEY_TEST_SPLIT) * np_data.shape[0]))
		train_d, train_l = np_data[:split, :], np_labels[:split, :]
		test_d, test_l = np_data[split:, :], np_labels[split:, :]
		return train_d, train_l, test_d, test_l

	@staticmethod
	def load_base_data(max_confession_length: int) -> Tuple[List[str], Tuple[Union[int, float], ...]]:
		"""
		loads the base data, which contains all the texts and
		all the labels.
		"""
		all_data = data.data_util.load_text_with_every_label("all_confessions/all", max_confession_length)
		texts, labels = zip(*[
			(row[data.data_util.CONFESSION_TEXT_INDEX], row[data.data_util.CONFESSION_REACTION_INDEX])
			for row in all_data
		])
		return texts, labels

	def train(self) -> None:
		"""
		trains self.model in place
		"""
		assert self.train_data is not None, \
			"test_data is None. The model may not have been trained. " \
			"Call the method `update_train_and_test_data` " \
			"on this instance of %s" % self.__class__.__name__
		history = self.model.fit(
			self.train_data,
			self.train_labels,
			epochs=self.get_hyperparam(BaseModel.KEY_EPOCHS),
			batch_size=self.get_hyperparam(BaseModel.KEY_BATCH_SIZE),
			validation_split=self.get_hyperparam(BaseModel.KEY_VALIDATION_SPLIT),
		)
		self.last_train_history = history

	def evaluate(self) -> Tuple[Any, Any]:
		""" evaluates the model and returns the metrics """
		assert self.test_data is not None, \
			"test_data is None. The model may not have been trained. " \
			"Call the method `update_train_and_test_data` " \
			"on this instance of %s" % self.__class__.__name__
		loss, metric = self.model.evaluate(self.test_data, self.test_labels)
		return loss, metric

	def predict(self, data_point: np.ndarray, *args, **kwargs) -> Any:
		return self.model.predict(data_point, args, kwargs)

	def save(self) -> None:
		"""
		saves the model into /models/storage with a name that uses all
		the model's hyperparameters
		"""
		models.storage.store.save_model(
			self.model,
			self.model_type_name,
			self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE),
			self.get_hyperparam(BaseModel.KEY_EPOCHS),
			self.get_hyperparam(BaseModel.KEY_BATCH_SIZE),
			self.get_hyperparam(BaseModel.KEY_VALIDATION_SPLIT),
		)

	def load(self) -> None:
		""" loads the model in place into self.model """
		model_path = models.storage.store.get_model_title(
			self.model_type_name,
			self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE),
			self.get_hyperparam(BaseModel.KEY_EPOCHS),
			self.get_hyperparam(BaseModel.KEY_BATCH_SIZE),
			self.get_hyperparam(BaseModel.KEY_VALIDATION_SPLIT),
		)
		self.model = keras.models.load_model(model_path)

	# **********************************
	# creates the model and evaluates it
	# **********************************

	def run(self, save: bool = True) -> None:
		"""
		runs the model
		"""
		self.setup_base_data()
		self.update_train_and_test_data()
		self.create()
		self.train()
		self.evaluate()
		if save:
			self.save()

	# ***************************************************
	# Methods to be implemented by all inheriting classes
	# ***************************************************

	def parse_base_data(
		self,
		sequences: List[List[int]],
		labels: Tuple[Union[int, float], ...]) -> Tuple[np.ndarray, np.ndarray]:
		"""
		given all the texts and labels, convert them into
		numpy arrays of training data and labels.
		use models.parsing.tokenizer.pad_data_sequences with
		sequences, self.word_to_index, and self.max_sequence_length
		todo - seems like most modifications are needed only on the labels
			-> so maybe just create a method that changes the labels or
			-> create a method that pads the sequences? however, the
			-> padding for lstm is different so we need to take that into
			-> consideration
		"""
		raise NotImplementedError

	def create(self) -> keras.Model:
		"""
		creates self.model in place
		use self.num_words and self.max_sequence_length
		"""
		raise NotImplementedError


if __name__ == '__main__':
	pass
