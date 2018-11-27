import numpy as np
import keras
import keras.layers
import keras.utils as ku
from models.base_model import BaseModel
import models.parsing.tokenizer
from typing import List, Tuple, Union


class LSTMGenerator(BaseModel):
    HIDDEN_NEURONS = 100

    def __init__(self, popularity_threshold: int = None):
        """
        Creates a binary classifier that maps to the index given in
        fb_reaction_index. this binary classifier will classify
        everything greater than cutoff as 1 and everything below
        as 0.
        """
        suffix = "" if popularity_threshold is None else "pt_%d" % popularity_threshold
        super(LSTMGenerator, self).__init__("lstm_generator%s" % suffix)
        self.popularity_threshold = popularity_threshold
        self.register_metadata("popularity_threshold")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 100)  # ~10m/epoch => ~16.67hrs
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 32)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 300)
        self.set_hyperparam(BaseModel.KEY_TEST_SPLIT, 0.0)
        self.set_hyperparam(BaseModel.KEY_VALIDATION_SPLIT, 0.05)
        self.set_hyperparam(BaseModel.KEY_MAX_CONFESSION_LENGTH, 600)

    def create(self) -> None:
        embedding_size = self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE)
        model = keras.Sequential()
        model.add(keras.layers.Embedding(
                self.num_words,
                embedding_size,
                embeddings_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),
                input_length=self.max_sequence_length - 1),
        )
        model.add(keras.layers.LSTM(LSTMGenerator.HIDDEN_NEURONS))
        model.add(keras.layers.Dense(self.num_words, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adamax", metrics=["accuracy"])
        model.summary()
        self.model = model

    def parse_base_data(
            self,
            sequences: List[List[int]],
            labels: List[Tuple[Union[int, float], ...]]) -> Tuple[np.ndarray, np.ndarray]:
        """ create new sequences (from subsequences) then create labels with them """
        sequences_over_threshold = \
            LSTMGenerator.get_sequences_over_threshold(sequences, labels, self.popularity_threshold)
        input_sequences = LSTMGenerator.get_lstm_sequences(sequences_over_threshold)
        padded = self.pad_sequences(input_sequences, padding="pre")
        train_data, new_labels = padded[:, :-1], padded[:, -1]
        new_labels = ku.to_categorical(new_labels, num_classes=self.num_words)
        return train_data, new_labels

    @staticmethod
    def get_sequences_over_threshold(
            sequences: List[List[int]],
            labels: List[Tuple[Union[int, float], ...]],
            threshold: int = None,
    ) -> List[List[int]]:
        """ return only the sequences that have more than {threshold} reactions """
        if threshold is None:
            return sequences
        reaction_counts = [sum(label) for label in labels]
        return [sequence for sequence, rxn_count in zip(sequences, reaction_counts) if rxn_count >= threshold]

    @staticmethod
    def get_lstm_sequences(sequences: List[List[int]]) -> List[List[int]]:
        """
        LSTMs need to always predict the next word, so we use this to cut each
        sequence into its sub-sequences so that our LSTM is able to learn
        faster.
        Ideally, we'd do this for all sub-sequences of length at least 2.
        However, that uses too much memory. So, instead we just do subsequences
        from index 0 till index 2-len(sequence) for each sequence
        """
        new_sequences = []
        for line in sequences:
            for i in range(2, len(line)):
                new_sequences.append(line[:i])
        print("done: creating LSTM %d sequences" % len(new_sequences))
        return new_sequences

    def generate(self, seed_text: str, total_word_additions: int):
        output_text = seed_text
        sequence_len = self.max_sequence_length - 1
        for index in range(total_word_additions):
            output_sequence = models.parsing.tokenizer.convert_text_to_sequence(
                    output_text,
                    self.word_to_index
            )
            padded_sequence = self.pad_sequences(
                    [output_sequence[-sequence_len:]],
                    maximum_sequence_len=sequence_len,
                    padding="pre"
            )
            # todo - pick randomly among the top 5 so that predictions are different every time
            predicted_index = self.model.predict_classes(padded_sequence, verbose=0)[0]
            output_text += " " + self.index_to_word[predicted_index]
        return output_text


if __name__ == "__main__":
    generator = LSTMGenerator(popularity_threshold=40)
    generator.run(save=True)
    # generator.load()
    examples = [("Anime girls are better than real girls who is with us anime lovers", 20)]
    for seed_text_, word_additions in examples:
        print(generator.generate(seed_text_, word_additions))
