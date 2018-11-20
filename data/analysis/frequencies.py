import json
from typing import Dict, List

import constants
import data.analysis.analysis_util as a_util


def get_frequencies(labels: List[int], fname: str, store: bool = False) -> Dict[int, int]:
    frequencies = a_util.get_frequency(labels)
    if store:
        with open("outputs/frequencies_%s.json" % fname, "w") as f:
            json.dump(frequencies, f)
            f.close()
    return frequencies


def get_label_frequencies(store: bool = False) -> List[Dict[int, int]]:
    likes, loves, wows, hahas, sads, angrys, comments = zip(*a_util.get_data_labels())
    reaction_names = ([r.lower() for r in constants.FB_REACTIONS] + ["comment"])
    reaction_frequencies_list = []
    for reaction, reaction_labels in zip(reaction_names, [likes, loves, wows, hahas, sads, angrys, comments]):
        fname = "%ss" % reaction
        frequencies = get_frequencies(reaction_labels, fname, store=store)
        reaction_frequencies_list.append(frequencies)
    return reaction_frequencies_list


def get_text_char_frequencies(store: bool = True) -> Dict[int, int]:
    char_counts = [len(text) for text in a_util.get_data_texts()]
    return get_frequencies(char_counts, "text_char_count", store=store)


if __name__ == '__main__':
    chars = get_text_char_frequencies(store=True)
