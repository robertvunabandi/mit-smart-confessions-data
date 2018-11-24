import json
import re
import os

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
	buckets: int) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
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
	:return : tuple<new_labels, buckets>
		-> new_labels : list[one_hot_label]
			-> one_hot_label : list[int]
				these are the labels where 1 means it's in
				bucket indicated by its index.
		-> buckets : list[bucket]
			-> bucket_range : tuple<min_value, max_value>
				-> min_value : int (exclusive)
				-> max_value : int (inclusive)
	"""
	assert len(labels) > buckets, \
		"number of labels must be greater than the number of buckets " \
		"otherwise this would be a one hot equivalent, which can be" \
		"with a faster algorithm."
	ordered_labels = sorted(labels)
	count_per_bucket = len(labels) // buckets
	bucket_map, last_point, current_bucket_number, count = {}, -1, 0, 0
	final_buckets = []
	for index, label in enumerate(ordered_labels):
		# create a new list every time because these objects are the
		# ones we will return. we don't want to cause any aliasing
		# error.
		bucket_map[label] = [1 if i == current_bucket_number else 0 for i in range(buckets)]
		count += 1
		next_label = ordered_labels[min(index + 1, len(ordered_labels) - 1)]
		if (count >= count_per_bucket) and (current_bucket_number < buckets - 1) and (label != next_label):
			count = 0
			current_bucket_number += 1
			final_buckets.append((last_point, label))
			last_point = label
	final_buckets.append((last_point, max(last_point, ordered_labels[-1])))
	output_labels = []
	for label in labels:
		output_labels.append(bucket_map[label])
	return output_labels, final_buckets


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
