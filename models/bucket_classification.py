from typing import List, Tuple, Union, Dict

import numpy as np
import keras
from keras import layers

import data.data_util
from models.base_model import BaseModel


class BucketClassification(BaseModel):
    ALL_REACTIONS_INDEX = -2
    ALL_REACTIONS_NO_COMMENTS_INDEX = -1
    # the following count were found through guess and check. They yield a
    # right balance of data points within the buckets, which is good for
    # training so that it doesn't keep predicting the same thing. By right
    # balance, I mean that not one of the buckets has most of the data
    # (i.e. it's not polarized).
    REACTION_INDEX_TO_OPTIMAL_BUCKET_COUNT = {
        -2: 24,  # all reactions including comments
        -1: 24,  # all reactions excluding comments
        data.data_util.FbReaction.LIKE_INDEX: 32,
        data.data_util.FbReaction.LOVE_INDEX: 13,
        data.data_util.FbReaction.WOW_INDEX: 8,
        data.data_util.FbReaction.HAHA_INDEX: 10,
        data.data_util.FbReaction.SAD_INDEX: 13,
        data.data_util.FbReaction.ANGRY_INDEX: 16,
        data.data_util.FbReaction.COMMENTS_INDEX: 8,
    }
    DEFAULT_BUCKET_COUNT = 15

    def __init__(self, fb_reaction_index: int, buckets: int = None, should_depolarize: bool = False):
        """
        Creates a binary classifier that maps to the index given in
        fb_reaction_index. this binary classifier will classify
        everything greater than cutoff as 1 and everything below
        as 0.
        """
        if buckets is None:
            buckets = BucketClassification.REACTION_INDEX_TO_OPTIMAL_BUCKET_COUNT.get(
                    fb_reaction_index,
                    BucketClassification.DEFAULT_BUCKET_COUNT
            )
        suffix = "%s_b%d" % ("all" if (fb_reaction_index == -1) else ("i%d" % fb_reaction_index), buckets)
        depol_suffix = "_depol" if should_depolarize else ""
        super(BucketClassification, self).__init__("bucket_classification_%s%s" % (suffix, depol_suffix))
        self.fb_reaction_index = fb_reaction_index
        self.buckets = buckets
        self.bucket_ranges = None
        self.bucket_frequencies = None
        self.register_metadata("fb_reaction_index")
        self.register_metadata("buckets")
        self.register_metadata("bucket_ranges")
        self.register_metadata("bucket_frequencies")
        self.set_hyperparam(BaseModel.KEY_EPOCHS, 75)
        self.set_hyperparam(BaseModel.KEY_BATCH_SIZE, 64)
        self.set_hyperparam(BaseModel.KEY_EMBEDDING_SIZE, 32)
        self.set_hyperparam(BaseModel.KEY_DEPOLARIZE, should_depolarize)

    def parse_base_data(
            self,
            sequences: List[List[int]],
            labels: List[Tuple[Union[int, float], ...]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        we maybe depolarize the data here because it can be very polarized.
        """
        if self.fb_reaction_index == -1:
            index_labels = [sum(label[:-1]) for label in labels]
        elif self.fb_reaction_index == -2:
            index_labels = [sum(label) for label in labels]
        else:
            index_labels = [label[self.fb_reaction_index] for label in labels]
        depol_sequences, depol_labels = BucketClassification.get_depolarized_data(
                sequences,
                index_labels,
                should_depolarize=self.get_hyperparam(BaseModel.KEY_DEPOLARIZE),
        )
        out = data.data_util.bucketize_labels(depol_labels, self.buckets)
        new_labels, self.bucket_ranges, b_frequencies = out
        freq_s = data.data_util.stringify_bucket_frequencies(b_frequencies)
        self.bucket_frequencies = BucketClassification.reformat_bucket_frequencies(b_frequencies)
        print("parse_base_data::Bucket Frequencies::\n%s" % freq_s)
        padded = self.pad_sequences(depol_sequences)
        return padded, np.array(new_labels)

    @staticmethod
    def get_depolarized_data(
            sequences: List[List[int]],
            labels: List[Union[int, float]],
            should_depolarize: bool) -> Tuple[List[List[int]], List[Union[int, float]]]:
        if not should_depolarize:
            return sequences, labels
        depol_label_with_indices = data.data_util.depolarize_data(labels)
        depol_sequences = [sequences[index] for index, _ in depol_label_with_indices]
        depol_labels = [labels[index] for index, _ in depol_label_with_indices]
        return depol_sequences, depol_labels

    @staticmethod
    def reformat_bucket_frequencies(
            b_frequencies: Dict[Tuple[Union[int, float], Union[int, float]], int]
    ) -> List[Tuple[Union[int, float], Union[int, float], int]]:
        """
        we need to reformat these because the keys are tuples, which is
        not the JSON standard. This throws an error when trying to save.
        """
        return sorted([k + (v,) for k, v in b_frequencies.items()], key=lambda freq: freq[0])

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
    label_index = 0
    bc = BucketClassification(
            label_index,
            should_depolarize=True
    )
    # bc.run(save=False)
    bc.load()
    for t, total_expected in [
        ("Today is my birthday and no one remembered.", 63),
        ("The SHPE president is soooo dreamy. Is he single?", 20),
        ("I have someone back home and here. I'm in love with both of them", 7),
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
