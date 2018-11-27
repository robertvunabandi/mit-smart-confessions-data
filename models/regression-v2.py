"""
************************
WE DO NOT USE THIS MODEL
************************
"""
import data.data_util
import numpy as np
import keras
from keras import layers
from models.base_model import BaseModel
from typing import List, Tuple, Union


class Regression(BaseModel):
    def __init__(self, fb_reaction_index: int = None):
        model_name = "regression_" + ("all" if fb_reaction_index is None else ("i%d" % fb_reaction_index))
        super(Regression, self).__init__(model_name)
        self.fb_reaction_index = fb_reaction_index
        self.avg: float = None
        self.std: float = None
        self.output_dim: int = None
        self.register_metadata("fb_reaction_index")
        self.register_metadata("avg")
        self.register_metadata("std")
        self.register_metadata("output_dim")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 100)
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 64)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 32)

    def parse_base_data(
            self,
            sequences: List[List[int]],
            labels: Tuple[Union[int, float], ...]) -> Tuple[np.ndarray, np.ndarray]:
        padded = self.pad_sequences(sequences)
        if self.fb_reaction_index is not None:
            self.output_dim = 1
            if self.fb_reaction_index == 7:
                np_labels = np.array([[sum(label[:6]) for label in labels]]).T
            else:
                np_labels = np.array([[label[self.fb_reaction_index] for label in labels]]).T
        else:
            self.output_dim = len(labels[0])
            np_labels = np.array(labels)
        np_labels, self.avg, self.std = data.data_util.standardize_array(np_labels)
        return padded, np_labels

    def create(self) -> None:
        embedding_size = self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE)
        model = keras.Sequential()
        model.add(layers.Embedding(self.num_words, embedding_size, input_length=self.max_sequence_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(self.output_dim, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.summary()
        self.model = model


if __name__ == '__main__':
    rm1 = Regression()
    rm1.run(save=False)
    # rm1.load()
    rm2 = Regression(fb_reaction_index=7)
    rm2.run(save=False)
    # rm2.load()
    for i, rm in enumerate([rm1, rm2]):
        print("RM #%d" % i)
        for t in [
            "I have romantic feelings for a close friend and I can't decide if it's better to tell them and risk damaging our friendship which I value a lot, or not tell them and hope the feelings fade over time",
            # total 54, comments 13
            "Everyone from tetazoo needs to grow up. Joke's on you guys when you can barely function in society.",
            # total 4, comments 3
            "Anyone who sings that country road or whatever the fuck that song is out loud at max volume deserves to be locked alone in a soundproof room for 24 hours and reflect on their life choices.",
            # total 145, comments 87
            "can someone please explain to me what a matrix is?"]:
            d = rm.convert_text_to_padded_sequence(t)
            pred = np.multiply(rm.predict(d), rm.std) + rm.avg
            print("text: %s" % t)
            if i == 0:
                print("like: %f, love: %f, wow: %f, haha: %f, sad: %f, angry: %f, comment: %f" % tuple(
                        pred.tolist()[0]))
            else:
                print("total: %f" % pred[0, 0])
            print()
