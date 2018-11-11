import data.data_util
import numpy as np
import keras
from keras import layers
from models.base_model import BaseModel
from typing import List, Tuple, Union


class BinaryClassification(BaseModel):

	def __init__(self, fb_reaction_index: int, cutoff: int = 10):
		"""
		Creates a binary classifier that maps to the index given in
		fb_reaction_index. this binary classifier will classify
		everything greater than cutoff as 1 and everything below
		as 0.
		"""
		super(BinaryClassification, self).__init__("bin_classification_i%d_c%d" % (fb_reaction_index, cutoff))
		self.fb_reaction_index = fb_reaction_index
		self.cutoff = cutoff
		self.register_metadata("fb_reaction_index")
		self.register_metadata("cutoff")
		self.set_hyperparam(BaseModel.KEY_EPOCHS, 150)
		self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 128)
		self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 32)

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
		model.add(layers.Flatten())
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
	for t in [
		"God damn it, I love you. Why do you have to be my ex's friend?",
		"I'm a sophomore and I still don't know what Taylor series are.",
		"I love MIT students",
		"Hello there",
	]:
		d = bc.convert_text_to_padded_sequence(t)
		print(t)
		print(bc.predict(d))
		print()
		# got these after training and predicting
		# [[0.00012126]  expected 0
		#  [0.99985516]  expected 1
		#  [0.01909499]
		#  [0.39393654]]
