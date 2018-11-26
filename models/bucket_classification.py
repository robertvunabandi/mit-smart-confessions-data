import data.data_util
import numpy as np
import keras
from keras import layers
from models.base_model import BaseModel
from typing import List, Tuple, Union


class BucketClassification(BaseModel):

    def __init__(self, fb_reaction_index: int, buckets: int = 25, should_depolarize: bool = False):
        """
        Creates a binary classifier that maps to the index given in
        fb_reaction_index. this binary classifier will classify
        everything greater than cutoff as 1 and everything below
        as 0.
        """
        suffix = "%s_b%d" % ("all" if (fb_reaction_index == -1) else ("i%d" % fb_reaction_index), buckets)
        depol_suffix = "_depol" if should_depolarize else ""
        super(BucketClassification, self).__init__("bucket_classification_%s%s" % (suffix, depol_suffix))
        self.fb_reaction_index = fb_reaction_index
        self.buckets = buckets
        self.bucket_ranges = None
        self.register_metadata("fb_reaction_index")
        self.register_metadata("buckets")
        self.register_metadata("bucket_ranges")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 50)
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 64)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 32)
        self.set_hyperparam(BaseModel.KEY_DEPOLARIZE, should_depolarize)

    def parse_base_data(
            self,
            sequences: List[List[int]],
            labels: List[Union[int, float]]) -> Tuple[np.ndarray, np.ndarray]:

        if self.fb_reaction_index != -1:
            index_labels = [label[self.fb_reaction_index] for label in labels]
        else:
            index_labels = [sum(label[:-1]) for label in labels]
        depol_label_with_indices = data.data_util.depolarize_data(index_labels)
        should_depolarize = self.get_hyperparam(BaseModel.KEY_DEPOLARIZE)
        depol_sequences = [sequences[index] for index, _ in depol_label_with_indices] \
            if should_depolarize \
            else sequences
        depol_labels = [index_labels[index] for index, _ in depol_label_with_indices] \
            if should_depolarize \
            else index_labels
        padded = self.pad_sequences(depol_sequences)
        new_labels, self.bucket_ranges, b_frequencies = data.data_util.bucketize_labels(
                depol_labels,
                self.buckets
        )
        print(
                "parse_base_data::Bucket Frequencies::\n\n"
                "%s" % data.data_util.stringify_bucket_frequencies(b_frequencies)
        )
        return padded, np.array(new_labels)

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
    bc = BucketClassification(data.data_util.FbReaction.LOVE_INDEX, buckets=25, should_depolarize=True)
    bc.run(save=False)
    # bc.load()
    for t, total_expected in [
        ("Today is my birthday and no one remembered.", 63),
        (
        "AAAAAAHHHHHHH I LOVE MY SO SO FREAKIN MUCH HE IS LITERALLY THE BEST WOW I NEED TO APPRECIATE HIM MORE AND BE NICER BC HE DESERVES ONLY THE VERY BEST!!!!!",
        13),
        ("The SHPE president is soooo dreamy. Is he single?", 20),
        (
        "to the girl in the gym the other eve that got really excited and ran over to the window and waved at one of the guys on the pool deck, that was super duper cute and i support yall",
        4),
        ("I have someone back home and here. I'm in love with both of them", 7),
        ("Mohammed Nasir and Ji Min Lee are the best most wholesome confessions commenters change my mind",
         19),
        (
        "I bet all the people who said “go vote!” before the midterm elections would regret encouraging me when they found out I voted republican.",
        46),
        ("where did daniel's hair go?!", 1),
        ("What if MIT Confessions is run by the GOD Eric Lander himself?? Function. Gene. Protein.", 55),
        ("Raise your hand if you've ever felt personally victimized by Ben Bitdiddle", 139),
    ]:
        d = bc.convert_text_to_padded_sequence(t)
        prediction = bc.predict(d)
        index_ = np.argmax(prediction, axis=1)[0]
        probs = [prediction[0, i] for i in
                 [max(0, index_ - 1), index_, min(index_ + 1, len(bc.bucket_ranges) - 1)]]
        bucket = bc.bucket_ranges[index_]
        print(t)
        print(probs, bucket, "(total expected %d)" % total_expected)
        print()
    # without depolarizing: loss: 0.0022 - acc: 0.9988 - val_loss: 0.1666 - val_acc: 0.9736
    # (-1.000, 0.000): 2784
    # (0.000, 1.000): 424
    # (1.000, 2.000): 205
    # (2.000, 4.000): 197
    # (4.000, 8.000): 164
    # (8.000, 46.000): 144
    # with depolarizing: loss: 0.0071 - acc: 0.9964 - val_loss: 0.0877 - val_acc: 0.9778
    # (-1.000, 0.000): 815
    # (0.000, 1.000): 507
    # (1.000, 2.000): 233
    # (2.000, 3.000): 267
    # (3.000, 4.000): 191
    # (4.000, 5.000): 173
    # (5.000, 7.000): 205
    # (7.000, 10.000): 152
    # (10.000, 13.000): 138
    # (13.000, 21.000): 124
    # (21.000, 46.000): 60
