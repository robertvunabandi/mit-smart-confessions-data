# type hints/annotation module
from typing import List, Tuple, Union, Any

import keras
import numpy as np

import data.data_util
import models.parsing.tokenizer
import models.storage.store


class BaseModel:
    # keys for hyperparameters
    KEY_TEST_SPLIT = "test_split"
    KEY_EPOCHS = "epochs"
    KEY_BATCH_SIZE = "batch_size"
    KEY_VALIDATION_SPLIT = "validation_split"
    KEY_EMBEDDING_SIZE = "embedding_size"
    KEY_MAX_CONFESSION_LENGTH = "max_confession_length"
    KEY_DATA_PATH = "data_path"
    KEY_DEPOLARIZE = "should_depolarize_data"
    DEFAULT_HYPERPARAMETER_VALUE_MAP = {
        "test_split": 0.20,
        "epochs": 100,
        "batch_size": 64,
        "validation_split": 0.33,
        "embedding_size": 16,
        "max_confession_length": 400,
        "data_path": "all_confessions/all",
        "should_depolarize_data": False,
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
        # Set the model's metadata (things are things that are needed
        # when predicting, which should also be loaded and not lost when
        # saving the model).
        self._metadata_attributes = [
            "max_sequence_length",
            "num_words",
            "word_to_index",
            "index_to_word",
            "last_train_history",
            "__hyperparams__"
        ]
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

    def register_metadata(self, metadata_attr: str):
        self._metadata_attributes.append(metadata_attr)

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
        np_data, np_labels = self.parse_base_data(self.sequences, self.labels)
        # shuffle the data
        np_data, np_labels = BaseModel.shuffle_data(np_data, np_labels)
        split = int(np_data.shape[0] - (self.get_hyperparam(BaseModel.KEY_TEST_SPLIT) * np_data.shape[0]))
        train_d, train_l = np_data[:split, :], np_labels[:split, :]
        test_d, test_l = np_data[split:, :], np_labels[split:, :]
        return train_d, train_l, test_d, test_l

    @staticmethod
    def shuffle_data(np_data: np.ndarray, np_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        shuffles the data and labels given, and returns the shuffled
        version in.
        """
        _, data_col_index = np_data.shape
        data_label_combination = np.hstack((np_data, np_labels))
        np.random.shuffle(data_label_combination)
        return data_label_combination[:, :data_col_index], data_label_combination[:, data_col_index:]

    def load_base_data(self, max_confession_length: int) -> Tuple[List[str], Tuple[Union[int, float], ...]]:
        """
        loads the base data, which contains all the texts and
        all the labels.
        """
        data_path = self.get_hyperparam(BaseModel.KEY_DATA_PATH)
        all_data = data.data_util.load_text_with_every_label(data_path, max_confession_length)
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
        self.last_train_history = history.history

    def evaluate(self) -> Tuple[Any, Any]:
        """ evaluates the model and returns the metrics """
        assert self.test_data is not None, \
            "test_data is None. The model may not have been trained. " \
            "Call the method `update_train_and_test_data` " \
            "on this instance of %s" % self.__class__.__name__
        evaluation = self.model.evaluate(self.test_data, self.test_labels)

        if len(evaluation) == 0:
            return None, None
        loss, metric = evaluation
        return loss, metric

    def predict(self, data_point: np.ndarray) -> Any:
        return self.model.predict(data_point)

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
        models.storage.store.save_model_metadata(
                self.get_model_metadata(),
                self.model_type_name,
                self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE),
                self.get_hyperparam(BaseModel.KEY_EPOCHS),
                self.get_hyperparam(BaseModel.KEY_BATCH_SIZE),
                self.get_hyperparam(BaseModel.KEY_VALIDATION_SPLIT),
        )

    def get_model_metadata(self) -> dict:
        """ get the model's metadata, specified with self.metadata_attributes """
        metadata = {
            "__attributes__": self._metadata_attributes,
            "__hyperparams__": self.hyperparams,
        }
        for attr in self._metadata_attributes:
            if attr == "__hyperparams__":
                continue
            metadata[attr] = getattr(self, attr, None)
        return metadata

    def load(self) -> None:
        """ loads the model in place into self.model """
        model_path = models.storage.store.get_model_title(
                self.model_type_name,
                self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE),
                self.get_hyperparam(BaseModel.KEY_EPOCHS),
                self.get_hyperparam(BaseModel.KEY_BATCH_SIZE),
                self.get_hyperparam(BaseModel.KEY_VALIDATION_SPLIT),
        )
        print("loading model from %s" % model_path)
        self.model = keras.models.load_model(model_path)
        model_metadata = models.storage.store.load_model_metadata(model_path)
        self._metadata_attributes = model_metadata["__attributes__"]
        self.set_model_metadata_attr(model_metadata)

    def set_model_metadata_attr(self, model_metadata: dict) -> None:
        """ set the model data attributes one by one """
        for attr in model_metadata["__attributes__"]:
            if attr == "__hyperparams__":
                # handle __hyperparams__ differently
                hyperparams = model_metadata.get(attr, {})
                for h_key, h_value in hyperparams.items():
                    self.set_hyperparam(h_key, h_value)
                continue
            if attr == "index_to_word":
                self.index_to_word = {
                    self.index_to_word[int(index)]: word
                    for index, word in model_metadata.get(attr, {}).items()
                }
                continue
            setattr(self, attr, model_metadata.get(attr, None))

    def pad_sequences(
            self,
            sequences: List[List[int]],
            maximum_sequence_len: int = None,
            padding: str = "post") -> np.ndarray:
        """
        pad the sequences with pad characters. If the maximum_sequence_len
        is not given, we use the default one that belongs to this class.
        Otherwise, we use what is provided. This maximum_sequence_len parameter
        is  necessary for RNN/LSTM models.
        """
        if maximum_sequence_len is None:
            maximum_sequence_len = self.max_sequence_length
        return models.parsing.tokenizer.pad_data_sequences(
                sequences,
                self.word_to_index,
                maximum_sequence_len,
                padding=padding
        )

    # **********************************
    # Creates the model and evaluates it
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

    # *****************************
    # Methods to test the model out
    # *****************************

    def convert_text_to_padded_sequence(self, text: str) -> np.ndarray:
        sequence = models.parsing.tokenizer.convert_text_to_sequence(text, self.word_to_index)
        return self.pad_sequences([sequence])

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
        use self.pad_sequences with sequences to pad
        """
        raise NotImplementedError

    def create(self) -> None:
        """
        creates self.model in place
        use self.num_words and self.max_sequence_length for embedding layer
        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
