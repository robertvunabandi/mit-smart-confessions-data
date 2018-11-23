import data.data_util
import numpy as np
import keras
from keras import layers, utils
from models.base_model import BaseModel
import models.parsing.tokenizer
from typing import List, Tuple, Union

class LSTMGenerator(BaseModel):
    #KEY_HIDDEN_NEURONS = "hidden_neurons"
    HIDDEN_NEURONS = 100

    def __init__(self):
        """
        Creates a binary classifier that maps to the index given in
        fb_reaction_index. this binary classifier will classify
        everything greater than cutoff as 1 and everything below
        as 0.
        """
        super(LSTMGenerator, self).__init__("lstm_generator")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 100)
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 128)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 16)
        self.set_hyperparam(BaseModel.KEY_TEST_SPLIT, 0)
        self.set_hyperparam(BaseModel.KEY_VALIDATION_SPLIT, 0)
        #self.set_hyperparam(LSTMGenerator.KEY_HIDDEN_NEURONS, 100)


    def create(self) -> None:
        embedding_size = self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE)
        hidden_neurons = LSTMGenerator.HIDDEN_NEURONS

        input_len = self.max_sequence_length - 1
        model = keras.Sequential()
        model.add(layers.Embedding(self.num_words, embedding_size, input_length=input_len))
        model.add(layers.LSTM(hidden_neurons))
        model.add(layers.Dense(self.num_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model


    def parse_base_data(self, sequences: List[List[int]],
        labels: Tuple[Union[int, float], ...]) -> Tuple[np.ndarray, np.ndarray]:

        input_sequences = []
        for line in sequences:
            for i in range(1, len(line)):
                n_gram_sequence = line[:i+1]
                input_sequences.append(n_gram_sequence)


        padded = self.pad_sequences(input_sequences, padding="pre")
        train_data, new_labels = padded[:,:-1],padded[:,-1]

        print(train_data)
        print(new_labels)

        new_labels = utils.to_categorical(new_labels, num_classes=self.num_words)

        return train_data, new_labels


    def generate(self, seed_text: str, total_word_additions: int):
        #output_sequence = self.convert_text_to_padded_sequence(seed_text)
        #initial_length = len(output_sequence)
        output_text = seed_text
        for index in range(total_word_additions):
            output_sequence = models.parsing.tokenizer.convert_text_to_sequence(output_text, self.word_to_index)
            padded_sequence = self.pad_sequences([output_sequence], maxlen=self.max_sequence_length - 1, padding="pre")
            predicted_index = self.model.predict_classes(padded_sequence, verbose=1)[0]
            output_text += " " + self.index_to_word[predicted_index]
            #output_sequence.append(predicted_index)
            #output_sequence = np.append(output_sequence, predicted_index)
            #padded_sequence[0, index] = predicted_index
        #index_to_word = dict([(index, word) for word, index in self.word_to_index.items()])
        #return models.parsing.tokenizer.convert_sequence_to_text(output_sequence, self.index_to_word)
        return output_text


if __name__ == '__main__':
    generator = LSTMGenerator()
    generator.run(save=True)
    #generator.load()
    examples = [("Anime girls are better than real girls who is with us anime lovers", 20)]
    for seed_text, total_word_additions in examples:
        print(generator.generate(seed_text, total_word_additions))






