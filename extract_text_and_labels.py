import json
import re


FB_REACTIONS = [fb_type.lower() for fb_type in ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]]
CONFESSION_ID_INDEX = 0
CONFESSION_TEXT_INDEX = 1
CONFESSION_REACTION_INDEX = 2


def load_json_data(name: str) -> list:
	with open("data/%s.json" % name, "r") as file:
		return json.load(file)


def extract_text_and_labels(feed: list) -> list:
	"""
	this returns, from each of the PostObject in the feed, a tuple
	such that the first item is the confession number, the second is
	the text, and the third are labels.

	:param feed : list[PostObject]
	:return : list[tuple<int, str, tuple[int]>]
	"""
	extracted = []
	for post_obj in feed:
		text = post_obj["message"]
		matches_confession_number = re.findall("^#\d+\s", text)
		if len(matches_confession_number) == 0:
			# skip, this is not a confession
			continue
		confession_number_string = matches_confession_number[0][1:]
		new_text = text[len(confession_number_string) + 1:]
		comment_count = post_obj.get("comments", 0)
		labels = tuple([post_obj.get("reactions", {}).get(fb_type, 0) for fb_type in FB_REACTIONS] + [comment_count])
		extracted.append((int(confession_number_string[:-1]), new_text, labels))
	return extracted


# example
if __name__ == '__main__':
	mit_confessions = load_json_data("mit_confessions_feed_array")
	data = extract_text_and_labels(mit_confessions)
	texts = [item[CONFESSION_TEXT_INDEX] for item in data]
