import numpy as np
import data.data_util
import keras
import models.parsing.tokenizer
from keras import layers
from models.base_model import BaseModel
from typing import List, Tuple, Union


class BinaryClassification(BaseModel):

	def __init__(self, fb_reaction_index: int, cutoff: int = 20):
		"""
		Creates a binary classifier that maps to the index given in
		fb_reaction_index. this binary classifier will classify
		everything greater than cutoff as 1 and everything below
		as 0.
		"""
		super(BinaryClassification, self).__init__("binary_classification")
		self.fb_reaction_index = fb_reaction_index
		self.cutoff = cutoff
		self.set_hyperparam(BaseModel.KEY_EPOCHS, 100)
		self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 128)

	def parse_base_data(
		self,
		sequences: List[List[int]],
		labels: Tuple[Union[int, float], ...]) -> Tuple[np.ndarray, np.ndarray]:
		padded = self.pad_sequences(sequences)
		labels = np.array([[label[self.fb_reaction_index] for label in labels]]).T
		labels[labels < self.cutoff] = 0
		labels[labels >= self.cutoff] = 1
		return padded, labels

	def create(self) -> None:
		embedding_size = self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE)
		model = keras.Sequential()
		model.add(layers.Embedding(self.num_words, embedding_size, input_length=self.max_sequence_length))
		model.add(layers.GlobalAveragePooling1D())
		model.add(layers.Dense(64, activation="relu"))
		model.add(layers.Dropout(0.25))
		model.add(layers.Dense(1, activation="sigmoid"))
		model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
		model.summary()
		self.model = model


if __name__ == '__main__':
	bc = BinaryClassification(data.data_util.FbReaction.LIKE_INDEX)
	# bc.run(save=True)
	bc.load()
	d = bc.convert_text_to_padded_sequence("God damn it, I love you. Why do you have to be my ex's friend?")
	print(bc.predict(d))
	# got [[0.00335187]] at running from scratch, got [[0.00335187]] at loading
