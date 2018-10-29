"""
See Copyright at the end of this file
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras


VOCAB_SIZE = 10000
PAD, START, UNKNOWN, UNUSED = "<PAD>", "<START>", "<UNK>", "<UNUSED>"
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_SIZE = 16


def get_imdb_word_index_maps(imdb_word_index: dict) -> tuple:
	# A dictionary mapping words to an integer index
	# The first indices are reserved
	word_to_index = {k: (v + 3) for k, v in imdb_word_index.items()}
	word_to_index[PAD] = 0
	word_to_index[START] = 1
	word_to_index[UNKNOWN] = 2
	word_to_index[UNUSED] = 3
	# create the reverse dictionary
	index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])
	return word_to_index, index_to_word


def decode_review(sequence: list, index_to_word: dict):
	return ' '.join([index_to_word.get(i, '?') for i in sequence])


def pad_sequences(*args, **kwargs):
	return keras.preprocessing.sequence.pad_sequences(*args, **kwargs)


def build_model():
	model = keras.Sequential()
	model.add(keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(32, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
	model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()
	return model


def run_imdb():
	imdb = keras.datasets.imdb
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)
	word_to_index, index_to_word = get_imdb_word_index_maps(imdb.get_word_index())
	train_data = pad_sequences(
		train_data,
		value=word_to_index[PAD],
		padding='post',
		maxlen=MAX_SEQUENCE_LENGTH
	)
	test_data = pad_sequences(test_data, value=word_to_index[PAD], padding='post', maxlen=MAX_SEQUENCE_LENGTH)
	model = build_model()
	x_val = train_data[:10000]
	partial_x_train = train_data[10000:]
	y_val = train_labels[:10000]
	partial_y_train = train_labels[10000:]
	print(partial_x_train.shape)
	print(partial_y_train.shape)
	history = model.fit(
		partial_x_train,
		partial_y_train,
		epochs=40,
		batch_size=512,
		validation_data=(x_val, y_val),
		verbose=1
	)
	results = model.evaluate(test_data, test_labels)
	print(results)


if __name__ == '__main__':
	run_imdb()

# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
