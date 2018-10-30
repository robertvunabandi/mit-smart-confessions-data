import json
from typing import Dict, Union, Tuple, Hashable, Any


CONF_FILES_TO_COMBINE = {
	"mit_confessions_feed_array": "mit_confessions",
	"summer_confessions_array": "mit_summer_confessions",
}
POST_OBJ_POST_ID = "post_id"
POST_OBJ_SOURCE_KEY = "source"
POST_OBJ_COMMENT_KEY = "comments"
POST_OBJ_REACTION_KEY = "reactions"
POST_OBJ_PAGE_DATA_KEY = "page_data"
KEYS = [
	"created_time",
	"message",
	POST_OBJ_POST_ID,
	POST_OBJ_COMMENT_KEY,
	POST_OBJ_REACTION_KEY,
	POST_OBJ_PAGE_DATA_KEY
]


def combine_data() -> None:
	all_data = []
	for file_name in CONF_FILES_TO_COMBINE.keys():
		with open("../%s.json" % file_name, "r") as file:
			data = json.load(file)
			all_data.extend(add_data_into_array(data, CONF_FILES_TO_COMBINE[file_name]))
	# this will need to be moved into all_confessions
	with open("../all.json", "w") as file:
		json.dump(all_data, file)
	print(
		"combined %d post objects from (%s)"
		% (len(all_data), ", ".join([file + ".json" for file in CONF_FILES_TO_COMBINE]))
	)


def add_data_into_array(data: list, source: str) -> list:
	return [
		add_to_dict({key: parse_key(post_obj, key) for key in KEYS}, (POST_OBJ_SOURCE_KEY, source))
		for post_obj in data
	]


def parse_key(post_obj: dict, key: str) -> Union[int, str, Dict[str, int], None]:
	if key == POST_OBJ_COMMENT_KEY:
		return int(post_obj.get(POST_OBJ_COMMENT_KEY, 0))
	if key == POST_OBJ_REACTION_KEY:
		reactions = {}
		post_obj_raw_reactions = post_obj.get(POST_OBJ_REACTION_KEY, {})
		for fb_reaction in post_obj_raw_reactions:
			reactions[fb_reaction] = int(post_obj_raw_reactions[fb_reaction])
		return reactions
	if key == POST_OBJ_PAGE_DATA_KEY:
		return post_obj.get(POST_OBJ_PAGE_DATA_KEY, None)
	if key == POST_OBJ_POST_ID:
		if post_obj.get(POST_OBJ_POST_ID, None) is None:
			return post_obj.get("id", None)
		return post_obj[POST_OBJ_POST_ID]
	return post_obj.get(key, None)


def add_to_dict(dic: dict, key_value_pair: Tuple[Hashable, Any]) -> dict:
	key, value = key_value_pair
	dic[key] = dic.get(key, value)
	return dic


if __name__ == '__main__':
	combine_data()
