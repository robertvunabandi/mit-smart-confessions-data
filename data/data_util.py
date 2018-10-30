import json
import re
import random
import utils
from constants import FB_REACTIONS
from typing import Tuple, List, Dict, Any, NewType


DEFAULT_COMMENT_COUNT = 0
CONFESSION_NUMBER_INDEX = 0
CONFESSION_TEXT_INDEX = 1
CONFESSION_REACTION_INDEX = 2


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
	label_index: int) -> Tuple[List[str], List[FbReactionCount]]:
	"""
	:param file_name : str
	:param label_index : str
	:return tuple<list[str], list[int]>
	"""
	data = list(load_text_with_every_label(file_name))
	random.shuffle(data)
	texts, labels = zip(*[
		(row[CONFESSION_TEXT_INDEX], row[CONFESSION_REACTION_INDEX][label_index])
		for row in data
	])
	return texts, labels


def load_text_with_every_label(
	file_name: str
) -> Tuple[List[str], List[Tuple[Likes, Loves, Wows, Hahas, Sads, Angrys, Comments]]]:
	data = _extract_text_and_labels(_load_text(file_name))
	random.shuffle(data)
	for row in data:
		yield (row[CONFESSION_NUMBER_INDEX], row[CONFESSION_TEXT_INDEX], row[CONFESSION_REACTION_INDEX])


# ************************************
# Private Methods To Be Used Here Only
# ************************************


def _load_text(file_name: str) -> List[Dict[str, Any]]:
	"""
	:param file_name : str
	:return list[dict<str, int|str>]
	"""
	with open("%s.json" % file_name, "r") as f:
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


if __name__ == '__main__':
    pass