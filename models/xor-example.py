"""
The XOR dataset splits data into 4 regions, 2 of which are
positive and 2 of which are negatives. Here's an example:
https://lh4.googleusercontent.com/-bqjMHgAMYGw/VLgVdaf--oI/AAAAAAAAAU0/Y13qf6ANe_k/s320/xor_data.png
Each region is a square of the same exact size. The four
regions are put together.
"""
import numpy as np
import keras.layers
import random
from keras.models import Model, Sequential
from typing import Tuple


XOR_INPUT_LENGTH = 2
XOR_NUMBER_OF_WORDS = 2
EMBEDDING_SIZE = 16
BASE_XOR_DATA = np.array([
	[1, 1],  # negative
	[-1, -1],  # negative
	[-1, 1],  # positive
	[1, -1],  # positive
])
BASE_XOR_LABELS = np.array([
	[0],
	[0],
	[1],
	[1],
])
XOR_FILE_PATH = "storage/xor.h5"


def get_random_xor_data(count: int, noise_frequency: float = 0.01):
	# declare bounds
	left, right, top, bottom, mid = -1, 1, -1, 1, 0
	noise_frequency = max(min(1.0, noise_frequency), 0.0)

	def get_label(point: Tuple[int, int]):
		x_, y_ = point
		is_top_left_quadrant = (x_ < mid) and (y_ > mid)
		is_bottom_right_quadrant = (x_ > mid) and (y_ < mid)
		return 1 if is_top_left_quadrant or is_bottom_right_quadrant else 0

	def random_coord(): return (random.random() * 2) - 1

	data, labels = [], []
	for _ in range(count):
		x, y = random_coord(), random_coord()
		label = get_label((x, y))
		data.append([x, y])
		# if noisy, make the label be opposite of what it is
		noisy = int(random.random() < noise_frequency)
		labels.append([label * (1 - noisy) + noisy * (1 - label)])
	return np.array(data), np.array(labels)


def create_model() -> Model:
	model = Sequential()
	model.add(keras.layers.Dense(100, activation="relu"))
	model.add(keras.layers.Dense(100, activation="relu"))
	model.add(keras.layers.Dense(100, activation="relu"))
	model.add(keras.layers.Dense(1, activation="sigmoid"))
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	return model


def run_xor(noise_frequency) -> Model:
	model = create_model()
	xor_data, xor_labels = get_random_xor_data(10000, noise_frequency)
	print(np.hstack((xor_data, xor_labels)))
	model.fit(
		xor_data,
		xor_labels,
		epochs=100,
		batch_size=64,
		validation_split=0.33,
		verbose=1,
	)
	model.save(XOR_FILE_PATH)
	loss, accuracy = model.evaluate(xor_data, xor_labels)
	print("loss: %f, accuracy: %f" % (loss, accuracy))
	predict_xor(model, BASE_XOR_DATA)
	return model


def predict_xor(model, data) -> None:
	print(np.round(model.predict(data).T, 6))


if __name__ == '__main__':
	run_xor(0.05)
	# model_ = keras.models.load_model(XOR_FILE_PATH)
	# predict_xor(model_, BASE_XOR_DATA)

"""
[[0.5890489  0.5862858  0.43531793 0.43531793]]
[[0.5890489  0.5862858  0.43531793 0.43531793]]

"""
