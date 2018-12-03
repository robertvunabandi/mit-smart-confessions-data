import os
import re
import json
import csv
import functools
from typing import List, Dict, Tuple, Any

import utils


ANALYSIS_DIR = os.path.dirname(__file__)
DATA_UTIL_DIR = ANALYSIS_DIR[:ANALYSIS_DIR.index("/analysis")]
FB_REACTIONS = [fb_type.lower() for fb_type in ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]]
DATA_KEYS = [
    ["created_time"],
    ["message"],
    ["post_id"],
    ["comments"],
    ["reactions", "like"],
    ["reactions", "love"],
    ["reactions", "wow"],
    ["reactions", "haha"],
    ["reactions", "sad"],
    ["reactions", "angry"],
    ["page_data", "page_fans"],
    ["page_data", "page_engaged_users"],
    ["page_data", "page_views"],
    ["page_data", "page_consumption"],
    ["source"],
]
SPECIAL_DEFAULT_FOR_KEY = {
    "created_time": "N/A",
    "message": "N/A",
    "post_id": "N/A",
    "page_data": "?",
}


def write_data_to_csv():
    data = load_data()
    # todo - should delete "all.csv" before writing into it
    with open("outputs/all.csv", "w") as f:
        csv_data_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_data_writer.writerow(get_data_columns())
        for index, data_point in enumerate(data):
            try:
                _, data_point["message"] = separate_confession_text_and_number(data_point["message"])
            except IndexError:
                # this will not work on non-confessions posts
                continue
            row = []
            for data_key in DATA_KEYS:
                row.append(get_data_point_column(data_key, data_point))
            row.append(len(data_point["message"]))
            csv_data_writer.writerow(row)
            if index % 1000 == 0:
                print("%d csv rows written" % index)
        print("%d csv rows written" % index)


def load_data():
    """
    :return list[dict<str, int|str>]
    """
    with open("%s/%s.json" % (DATA_UTIL_DIR, "all_confessions/all"), "r") as f:
        return json.load(f)


def get_data_columns():
    return [functools.reduce(lambda _, new_key: new_key, data_key) for data_key in DATA_KEYS] \
           + ["character count"]


def separate_confession_text_and_number(raw_text: str) -> Tuple[int, str]:
    # match text that start with a "#" and numbers followed by a space
    matches_confession_number = re.findall("^#\d+\s", raw_text)
    confession_number_string = matches_confession_number[0][1:]
    confession_number = int(confession_number_string[:-1])
    return confession_number, utils.Str.remove_whitespaces(raw_text[len(confession_number_string) + 1:])


def get_data_point_column(data_key: List[str], data_point: Dict[str, Any]):
    default_value = SPECIAL_DEFAULT_FOR_KEY.get(data_key[0], 0)
    try:
        output = functools.reduce(
                lambda before, new_key: {} if before.get(new_key, {}) is None else before.get(new_key, {}),
                data_key,
                data_point
        )
    except Exception as e:
        print(data_key)
        print(data_point)
        raise e
    if type(output) == dict:
        return default_value
    return output


if __name__ == '__main__':
    write_data_to_csv()
