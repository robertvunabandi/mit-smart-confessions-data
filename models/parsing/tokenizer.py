from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from typing import Dict, List, Tuple, Any
import numpy as np
import utils


# words to filter out of the strings, these are base filters which
# are updated below
FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
PUNCTUATION_FILTERS = "!?.,;"
NON_FILTERS = "-_"
# remove punctuation and non-filters from the base filters
FILTERS = utils.Str.remove_chars_from_string(FILTERS, PUNCTUATION_FILTERS)
FILTERS = utils.Str.remove_chars_from_string(FILTERS, NON_FILTERS)
# additional characters we need in our vocabulary
PAD, START, UNKNOWN, UNUSED, END = "<PAD>", "<START>", "<UNK>", "<UNUSED>", "<END>"
# todo - figure out whether we want/need to use this or not
# additional words to add to the tokenizer
# the Tokenizer object will ignore the words that it doesn't know
# when making predictions. adding these words can help the tokenizer
# recognize them. add each words to this list
ADDITIONAL_WORDS = []


# ******************************************
# Main Methods:
# Use these to tokenize text, split datasets
# into test and train, and to convert text
# into sequences and vice versa
# ******************************************


def get_text_items(sentences: List[str]) -> Tuple[List[List[int]], int, Dict[int, str], Dict[str, int]]:
    """
    :param sentences : list[str]
    :return Tuple<List[List[int]], int, Dict[int, str], Dict[str, int]>
    """
    tokenizer = Tokenizer(filters=FILTERS, split=" ")
    new_sentences = [
        utils.Str.put_spaces_around_punctuations(sentence, char_punctuations=PUNCTUATION_FILTERS)
        for sentence in sentences
    ]
    tokenizer.fit_on_texts(list(new_sentences) + ["".join(ADDITIONAL_WORDS)])
    index_to_word, word_to_index = _word_index_maps(tokenizer)
    num_words = len(word_to_index)
    sequences = _update_sequences_with_new_word_index_map(
            tokenizer.texts_to_sequences(new_sentences),
            tokenizer.index_word,
            word_to_index,
    )
    return sequences, num_words, index_to_word, word_to_index


def split_dataset(
        sequences: List[List[int]],
        labels: List[int] or Tuple[int, ...],
        word_to_index: Dict[str, int],
        test_percentage: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    :param sequences : list[list[int]]
    :param labels : list[int]|tuple[int]
    :param word_to_index : dict<str, int>
    :param test_percentage : float
        of all the data, how many are used for the test set
    :return : tuple<np.array, np.array, np.array, np.array, int>
        -> training data, training labels, testing data, testing labels, max sequence length
    """
    max_sequence_length = find_max_sentence_length(sequences)
    split_index = len(sequences) - int(test_percentage * len(sequences))
    raw_train_data, raw_train_labels = sequences[:split_index], list(labels[:split_index])
    raw_test_data, raw_test_labels = sequences[split_index:], list(labels[split_index:])
    train_data = pad_data_sequences(raw_train_data, word_to_index, max_sequence_length)
    test_data = pad_data_sequences(raw_test_data, word_to_index, max_sequence_length)
    if len(np.array(labels).shape) == 1:
        train_labels = np.array([raw_train_labels]).T
        test_labels = np.array([raw_test_labels]).T
    else:
        train_labels = np.array(raw_train_labels)
        test_labels = np.array(raw_test_labels)
    return train_data, train_labels, test_data, test_labels, max_sequence_length


def find_max_sentence_length(sequences: list) -> int:
    """
    returns the longest sequence, which will be used as the maximum sequence.
    :param sequences : list[list[int]]
    :return int
    """
    max_length = None
    for seq in sequences:
        if max_length is None or len(seq) > max_length:
            max_length = len(seq)
    return max_length


def pad_data_sequences(
        sequences: List[List[int]],
        word_to_index: Dict[str, int],
        max_sequence_length: int,
        padding: str = "post") -> np.ndarray:
    return pad_sequences(
            sequences, value=word_to_index[PAD], maxlen=max_sequence_length, padding=padding
    )


def convert_text_to_sequence(text: str, word_to_index: Dict[str, int]) -> list:
    """
    :param text : str
    :param word_to_index : dict<str, int>
    :return list[int]
    """
    punctuated_sentence = utils.Str.put_spaces_around_punctuations(text, char_punctuations=PUNCTUATION_FILTERS)
    words = utils.Str.extract_words_from_text_with_filters(punctuated_sentence, FILTERS)
    return [word_to_index.get(word, word_to_index[UNKNOWN]) for word in words]


def convert_sequence_to_text(sequence: List[int], index_to_word: Dict[int, str]) -> str:
    """
    :param sequence : list[int]
    :param index_to_word : dict<int, str>
    :return str
    """
    output_text = " ".join([index_to_word.get(index, UNKNOWN) for index in sequence])
    return utils.Str.remove_spaces_around_punctuations(output_text, PUNCTUATION_FILTERS)


# ************************************
# Private Methods To Be Used Here Only
# ************************************

def _word_index_maps(tokenizer: Tokenizer) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    return dictionaries that can be used to map indices
    to words and vice versa
    :param tokenizer : Tokenizer
    :return tuple<dict<int, str>, dict<str, int>>
    """
    word_to_index = {k: (v + 4) for k, v in tokenizer.word_index.items()}
    word_to_index[START] = 0
    word_to_index[PAD] = 1
    word_to_index[UNKNOWN] = 2
    word_to_index[UNUSED] = 3
    word_to_index[END] = 4
    index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])
    return index_to_word, word_to_index


def _update_sequences_with_new_word_index_map(
        raw_sequences: List[List[int]],
        raw_index_to_word: Dict[int, str],
        word_to_index: Dict[str, int]) -> List[List[int]]:
    """
    Converts the raw sequences to a sequence with an updated word_to_index
    dictionary. To do that, we need the original index_to_word, which is
    given in raw_index_to_word.
    """
    return [
        [word_to_index[raw_index_to_word[index]] for index in sequence]
        for sequence in raw_sequences
    ]
