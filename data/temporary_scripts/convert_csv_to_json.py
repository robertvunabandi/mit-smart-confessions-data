"""
This was used to convert summer confessions from csv to json because
json is much easier to work with and key.
"""

import csv
import re
import json


SUMMER_CONFESSIONS_CSV_ROWS = [
	"status_id",
	"status_message",
	"link_name",
	"status_type",
	"status_link",
	"permalink_url",
	"status_published",
	"num_reactions",
	"num_comments",
	"num_shares",
	"num_likes",
	"num_loves",
	"num_wows",
	"num_hahas",
	"num_sads",
	"num_angrys",
]

SUMMER_CONFESSIONS_CSV_ROWS_MAP = {
	"status_id": "id",
	"status_message": "message",
	"num_comments": "comments",
	"num_likes": "like",
	"num_loves": "love",
	"num_wows": "wow",
	"num_hahas": "haha",
	"num_sads": "sad",
	"num_angrys": "angry",
	# we don't care about the rest here
	"link_name": "link_name",
	"status_type": "status_type",
	"status_link": "status_link",
	"permalink_url": "permalink_url",
	"status_published": "status_published",
	"num_reactions": "num_reactions",
	"num_shares": "num_shares",
}


def is_facebook_reaction(field: str) -> bool:
	return bool(re.match("(like|love|wow|haha|sad|angry)", field))


def should_include_field_in_confession(field: str) -> bool:
	return bool(re.match("(id|message|story|comments)", field)) or is_facebook_reaction(field)


def summer_confessions() -> list:
	data = []
	with open("files/summer_confessions.csv", "r") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			confession = {}
			for field in SUMMER_CONFESSIONS_CSV_ROWS:
				confession_field = SUMMER_CONFESSIONS_CSV_ROWS_MAP[field]
				if is_facebook_reaction(confession_field):
					confession.setdefault("reactions", {})[confession_field] = row[field]
				elif should_include_field_in_confession(confession_field):
					confession[confession_field] = row[field]
			data.append(confession)
	return data


if __name__ == '__main__':
	sc = summer_confessions()
	with open("files/summer_confessions.json", "w") as json_file:
		json.dump(sc, json_file)
