import json
import re
import os
import random
import math

from typing import Tuple, List, Dict, Any, NewType, Union
import numpy as np

from constants import FB_REACTIONS
import utils


DATA_UTIL_DIR = os.path.dirname(__file__)
DEFAULT_COMMENT_COUNT = 0
CONFESSION_NUMBER_INDEX = 0
CONFESSION_TEXT_INDEX = 1
CONFESSION_REACTION_INDEX = 2
MAX_CONFESSION_CHARACTER_LENGTH = 500


class FbReaction:
    LIKE_INDEX: int = 0
    LOVE_INDEX: int = 1
    WOW_INDEX: int = 2
    HAHA_INDEX: int = 3
    SAD_INDEX: int = 4
    ANGRY_INDEX: int = 5
    COMMENTS_INDEX: int = 6


# some custom types for code readability
FbReactionCount = NewType("FbReactionCount", Union[int, float])


# ****************************************
# Main Methods: Use These to Load Examples
# ****************************************


def load_text_with_specific_label(
        file_name: str,
        label_index: int,
        max_conf_char_length: int = None) -> Tuple[List[str], List[FbReactionCount]]:
    """
    loads the text like in load_text_with_every_label, but we only limit it
    to one set of labels instead.
    :return tuple<confessions, labels>
        -> confessions: list[str]
        -> labels: list[int]
    """
    data = load_text_with_every_label(file_name, max_conf_char_length)
    texts, labels = zip(*[
        (row[CONFESSION_TEXT_INDEX], row[CONFESSION_REACTION_INDEX][label_index])
        for row in data
    ])
    return texts, labels


def load_text_with_labels_percentages(
        file_name: str,
        max_conf_char_length: int = None) -> Tuple[List[str], List[FbReactionCount]]:
    """
    :return tuple<confessions, labels>
        -> confessions : list[str]
        -> labels : list[tuple<reaction_percentage x 6, total_reactions>]
            -> reaction_percentage : float
            -> total_reactions : int (the total number of reaction for that confession)
    """
    data = load_text_with_every_label(file_name, max_conf_char_length)
    texts, labels = zip(*[
        (row[CONFESSION_TEXT_INDEX], _labels_to_percentages(row[CONFESSION_REACTION_INDEX]))
        for row in data
    ])
    return texts, labels


def load_text_with_every_label(
        file_name: str,
        max_confession_character_length: int = None,
) -> List[Tuple[int, str, Tuple[FbReactionCount, ...]]]:
    """
    loads all the data from the json data file
    :param file_name : str
    :param max_confession_character_length : int
    :return List[Tuple[confession_id, confession_text, confession_labels]]
        -> confession_id : int
        -> confession_text : str
        -> confession_labels : Tuple[int, ...]
    """
    data = _extract_text_and_labels(_load_text(file_name))
    dataset = []
    max_char_length_is_none = max_confession_character_length is None
    for row in data:
        if max_char_length_is_none or len(row[CONFESSION_TEXT_INDEX]) < max_confession_character_length:
            dataset.append(
                    (row[CONFESSION_NUMBER_INDEX], row[CONFESSION_TEXT_INDEX], row[CONFESSION_REACTION_INDEX])
            )
    return dataset


# ******************************
# Methods to Manipulate the Data
# ******************************

def bucketize_labels(
        labels: List[int or float],
        buckets: int) -> Tuple[List[List[int]], List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    bucketize the labels with the number of buckets such that
    each bucket has a roughly equal number of labels in it. I.e.,
    for each label i, return a tuple of the form (0,0,...,0,1,0,...,0)
    with length {buckets} (a one hot encoding denoting which bucket
    this label belongs to).

    In addition to that, return a tuple of ranges that determine what
    are the bucket ranges.

    :param labels : list[int|float]
        labels, which are numbers that may be floating points
    :param buckets : int
    :return : tuple<new_labels, buckets, bucket_frequencies>
        -> new_labels : list[one_hot_label]
            -> one_hot_label : list[int]
                these are the labels where 1 means it's in
                bucket indicated by its index.
        -> buckets : list[bucket]
            -> bucket_range : tuple<min_value, max_value>
                -> min_value : int (exclusive)
                -> max_value : int (inclusive)
        -> bucket_frequencies : dict[tuple, int]
            how many labels we have in the bucket
    """
    assert len(labels) > buckets, \
        "number of labels must be greater than the number of buckets " \
        "otherwise this would be a one hot equivalent, which can be" \
        "with a faster algorithm."
    ordered_labels = sorted(labels)
    count_per_bucket = len(labels) // buckets
    bucket_map, last_point, current_bucket_number, count = {}, -1, 0, 0
    final_buckets = []
    bucket_frequencies = {}
    for index, label in enumerate(ordered_labels):
        # avoid aliasing bugs by creating a new list every time
        bucket_map[label] = [1 if i == current_bucket_number else 0 for i in range(buckets)]
        count += 1
        next_label = ordered_labels[min(index + 1, len(ordered_labels) - 1)]
        # for each of the if statements below:
        # 1: we want to stop when we reach the maximum count per bucket
        # 2: we don't want to include the same label in the same bucket
        # 3: we don't want to increase the count if we reach the last bucket
        if (count >= count_per_bucket) and (label != next_label) and (current_bucket_number < buckets - 1):
            bucket_frequencies[(last_point, label)] = count
            count = 0
            current_bucket_number += 1
            final_buckets.append((last_point, label))
            last_point = label
    # do the last bucket
    last_bucket = (last_point, max(last_point, ordered_labels[-1]))
    final_buckets.append(last_bucket)
    bucket_frequencies[last_bucket] = count
    output_labels = []
    for label in labels:
        output_labels.append(bucket_map[label])
    return output_labels, final_buckets, bucket_frequencies


def stringify_bucket_frequencies(bucket_frequencies: Dict[Tuple[int, int], int]) -> str:
    return "\n".join([
        "(%.3f, %.3f): %d" % (range_min, range_max, bucket_frequencies[(range_min, range_max)])
        for range_min, range_max in sorted(bucket_frequencies.keys(), key=lambda buck_range: buck_range[0])
    ])


def standardize_array(
        array: Union[List[int], Tuple[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param array : list[int] | tuple[int] | np.array
    :return tuple<np.array, np.array, np.array>
        -> standardized array, average, standard deviation
    """
    if type(array) == list or type(array) == tuple:
        return standardize_array(np.array(array))
    if isinstance(array, np.ndarray):
        avg, std = np.average(array, axis=0), np.std(array, axis=0)
        return (array - avg) / std, avg, std
    raise TypeError("array must be either a list of numbers or a numpy array")


def depolarize_data(labels: List[Union[int, float]]) -> List[Tuple[int, Union[int, float]]]:
    """
    systematically depolarize the data by slightly reducing the labels that
    are very polarized and slightly duplicating the sparse ones.
    """
    label_multipliers = _get_depolarizing_multiplier_map(labels, max_prob_threshold=0.5)
    new_labels = []
    for index, label in enumerate(labels):
        multiplier = label_multipliers[label]
        if multiplier == 1.0:
            new_labels.append((index, label))
            continue
        prob = multiplier % 1.0
        # we need to iterate at least twice because this led to the data
        # not being depolarized in such a way that it doesn't look like
        # what it was before. also, x // 1 = 0 for all x in [0, 1)
        for _ in range(max(int(multiplier // 1), 2)):
            if random.random() < prob:
                new_labels.append((index, label))
    return new_labels


# ************************************
# Private Methods To Be Used Here Only
# ************************************


def _labels_to_percentages(
        labels: Tuple[FbReactionCount, ...]) -> Tuple[float, ...]:
    label_percentage = [0.0] * 7
    total = sum(labels[:-1])  # don't include comments
    if total == 0.0:
        return tuple(label_percentage)
    for index in range(6):
        label_percentage[index] = labels[index] / total
    label_percentage[-1] = total
    return tuple(label_percentage)


def _load_text(file_name: str) -> List[Dict[str, Any]]:
    """
    :param file_name : str
    :return list[dict<str, int|str>]
    """
    with open("%s/%s.json" % (DATA_UTIL_DIR, file_name), "r") as f:
        return json.load(f)


def _extract_text_and_labels(feed: list) -> List[Tuple[int, str, Tuple[FbReactionCount, ...]]]:
    """
    :param feed : list[PostObject]
    :return : list[tuple<int, str, tuple[int]>]
        -> (confession_number, text, labels)
    """
    extracted = []
    for post_obj in feed:
        raw_text = post_obj["message"]
        # match text that start with a "#" and numbers followed by a space
        matches_confession_number = re.findall("^#\d+\s", raw_text)
        # if no match, skip: this is not a confession, it's a page post
        if len(matches_confession_number) == 0:
            continue
        confession_number_string = matches_confession_number[0][1:]
        confession_number = int(confession_number_string[:-1])
        text = utils.Str.remove_whitespaces(raw_text[len(confession_number_string) + 1:])
        labels = _get_labels(post_obj)
        extracted.append((confession_number, text, labels))
    return extracted


def _get_labels(post_obj: dict) -> Tuple[FbReactionCount, ...]:
    """
    :param post_obj : dict<str, T>
    :return tuple<int, int, int, int, int, int, int>
        -> (Each reactions x 6, comment_count)
    """
    comment_count = post_obj.get("comments", DEFAULT_COMMENT_COUNT)
    return tuple(
            [post_obj.get("reactions", {}).get(fb_type, 0) for fb_type in FB_REACTIONS] + [comment_count]
    )


def _get_depolarizing_multiplier_map(
        labels: List[Union[int, float]],
        max_prob_threshold: float = 0.5) -> Dict[Union[int, float], float]:
    """
    for each label, get a multiplier such that we get a multiplier. this
    number is a number such that multiplier % 1 gives us a probability
    and for each multiplier // 1, we duplicate the data.
    :return : list[tuple<label, multiplier>]
        -> multiplier: float (in range [0, infinity])
    """
    label_dist = _get_label_distribution(labels)
    max_prob = max(label_dist.values())
    # we do not want to depolarize the data if the data is not polar.
    # returning a multiplier of 1 means we're allowing all the data
    # to just be represented.
    if max_prob <= max_prob_threshold:
        return {label: 1 for label, prob in label_dist.items()}
    return {label: _get_depolarizing_multiplier(prob, max_prob) for label, prob in label_dist.items()}


def _get_label_distribution(labels: List[Union[int, float]]) -> Dict[Union[int, float], float]:
    """
    returns the distribution of the labels as a key value pair
    where the key is the label and the value is the distribution
    """
    count_map, total = {}, 0
    for label in labels:
        count_map[label], total = count_map.get(label, 0) + 1, total + 1
    dist = {}
    for label, count in count_map.items():
        dist[label] = (count / total)
    return dist


def _get_depolarizing_multiplier(prob: float, max_prob: float) -> float:
    """
    The functions below are chosen in such a way that low values of of prob
    get a high multiplier and high values of prob get a low multiplier. This
    multiplier scales with how much a data is multiplied (or lost).
    This is hard to explain, please see the desmos graph:
    https://www.desmos.com/calculator/q4d9f0gcyr
    """
    if max_prob <= 0.5:
        return 1.0
    f = lambda x: (-0.5 * (max_prob ** 4) * ((x + 1) ** 2)) + 2.14 * max_prob
    g = lambda x: 0.5 * math.e ** ((1 + max_prob) ** 1.4 - ((15/max_prob**2) * x ** 2))
    return f(prob) + g(prob) * (1 if max_prob > 0.65 else 0)


if __name__ == '__main__':
    all_labels = [
        label[CONFESSION_REACTION_INDEX] for label in
        load_text_with_every_label("all_confessions/all")
    ]
    labels_list = [[label[label_index] for label in all_labels] for label_index in range(7)]
    for labels_ in labels_list:
        print("-----")
        print("before")
        print("COUNT: %d" % len(labels_))
        before_labels = sorted([
            (label, round(prob, 3))
            for label, prob in _get_label_distribution(labels_).items()
        ], key=lambda tup: tup[0])
        print(before_labels)
        print("after")
        new_labels_ = [label for index, label in depolarize_data(labels_)]
        print("COUNT: %d" % len(new_labels_))
        after_labels = sorted([
            (label, round(prob, 3))
            for label, prob in _get_label_distribution(new_labels_).items()
        ], key=lambda tup: tup[0])
        print(after_labels)
