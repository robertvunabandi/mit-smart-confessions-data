import data.data_util
import numpy as np
import keras
from keras import layers
from models.base_model import BaseModel
from typing import List, Tuple, Union


class BucketClassification(BaseModel):

    def __init__(self, fb_reaction_index: int, buckets: int = 25):
        """
        Creates a binary classifier that maps to the index given in
        fb_reaction_index. this binary classifier will classify
        everything greater than cutoff as 1 and everything below
        as 0.
        """
        model_name = "bucket_classification_i%d_b%d" % (fb_reaction_index, buckets)
        super(BucketClassification, self).__init__(model_name)
        self.fb_reaction_index = fb_reaction_index
        self.buckets = buckets
        self.bucket_ranges = None
        self.register_metadata("fb_reaction_index")
        self.register_metadata("buckets")
        self.register_metadata("bucket_ranges")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 50)
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 64)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 32)

    def parse_base_data(
            self,
            sequences: List[List[int]],
            labels: Tuple[Union[int, float], ...]) -> Tuple[np.ndarray, np.ndarray]:
        padded = self.pad_sequences(sequences)
        if self.fb_reaction_index != -1:
            index_labels = [label[self.fb_reaction_index] for label in labels]
        else:
            index_labels = [sum(label[:-1]) for label in labels]
        new_labels, self.bucket_ranges = data.data_util.bucketize_labels(index_labels, self.buckets)
        return padded,  np.array(new_labels)

    def create(self) -> None:
        embedding_size = self.get_hyperparam(BaseModel.KEY_EMBEDDING_SIZE)
        model = keras.Sequential()
        model.add(layers.Embedding(self.num_words, embedding_size, input_length=self.max_sequence_length))
        model.add(layers.Conv1D(64, kernel_size=10, strides=1, padding='valid', activation="relu"))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.1))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(self.buckets, activation="softmax"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        self.model = model


if __name__ == '__main__':
    bc = BucketClassification(-1)
    # bc.run(save=True)
    bc.load()
    for t, expected in [
        ("Today is my birthday and no one remembered.", 63),
        ("AAAAAAHHHHHHH I LOVE MY SO SO FREAKIN MUCH HE IS LITERALLY THE BEST WOW I NEED TO APPRECIATE HIM MORE AND BE NICER BC HE DESERVES ONLY THE VERY BEST!!!!!", 13),
        ("The SHPE president is soooo dreamy. Is he single?", 20),
        ("to the girl in the gym the other eve that got really excited and ran over to the window and waved at one of the guys on the pool deck, that was super duper cute and i support yall", 4),
        ("I have someone back home and here. I'm in love with both of them", 7),
        ("Mohammed Nasir and Ji Min Lee are the best most wholesome confessions commenters change my mind", 19),
        ("I bet all the people who said “go vote!” before the midterm elections would regret encouraging me when they found out I voted republican.", 46),
        ("where did daniel's hair go?!", 1),
        ("What if MIT Confessions is run by the GOD Eric Lander himself?? Function. Gene. Protein.", 55),
        ("Raise your hand if you've ever felt personally victimized by Ben Bitdiddle", 139),
    ]:

        d = bc.convert_text_to_padded_sequence(t)
        prediction = bc.predict(d)
        index = np.argmax(prediction, axis=1)[0]
        probs = [prediction[0, i] for i in [max(0, index - 1), index, min(index + 1, len(bc.bucket_ranges) - 1)]]
        bucket = bc.bucket_ranges[index]
        print(t)
        print(probs, bucket, "(expected %d)" % expected)
        print()
    # got these after training and predicting
    # [[0.02407354]  expected 0
    #  [0.9994462]   expected 1
    #  [0.8864635]
    #  [0.5478533]]
