import json
import re
import random
import utils
import os
from constants import FB_REACTIONS
from typing import Tuple, List, Dict, Any, NewType, Union
import numpy as np


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
FbReactionCount = NewType("FbReactionCount", int)
Likes = NewType("Likes", FbReactionCount)
Loves = NewType("Loves", FbReactionCount)
Wows = NewType("Wows", FbReactionCount)
Hahas = NewType("Hahas", FbReactionCount)
Sads = NewType("Sads", FbReactionCount)
Angrys = NewType("Angrys", FbReactionCount)
Comments = NewType("Comments", FbReactionCount)


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
	labels: List[int],
	buckets: int) -> Tuple[List[Tuple[int, ...]], Tuple[Tuple[int, int], ...]]:
	"""
	todo - method to be implemented
	bucketize the labels with the number of buckets such that
	each bucket has a roughly equal number of labels in it. I.e.,
	for each label i, return a tuple of the form (0,0,...,0,1,0,...,0)
	with length {buckets} (a one hot encoding denoting which bucket
	this label belongs to).

	In addition to that, return a tuple of ranges that determine what
	are the bucket ranges.

	:param labels : list[int]
	:param buckets : int
	:return : tuple<labels_in_buckets, bucket_ranges>
		-> labels_in_buckets : list[bucket_for_label]
			-> bucket_for_label : tuple[int]
		-> bucket_ranges : Tuple[bucket_range]
			-> bucket_range : Tuple<min_value, max_value>
				-> min_value : int
				-> max_value : int
	"""
	raise NotImplementedError


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
