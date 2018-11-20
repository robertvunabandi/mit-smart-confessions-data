import json
from typing import List, Tuple, Dict

import constants
import data.analysis.analysis_util as a_util


# adding comments to the number of reactions
N_REACTIONS = len(constants.FB_REACTIONS) + 1
N_BINS = 50
REACTION_NAMES = ([r.lower() for r in constants.FB_REACTIONS] + ["comment"])


def plot_histograms(n_bins: int = N_BINS) -> None:
    # todo - set the max to be at like 100...
    # todo - remove % plot because it's useless
    labels = a_util.get_data_labels()
    label_lists = [[] for _ in range(N_REACTIONS)]
    for label in labels:
        for index, label_list in enumerate(label_lists):
            label_list.append(label[index])
    for reaction, reaction_labels in zip(REACTION_NAMES, label_lists):
        hist = a_util.Histogram(reaction_labels, n_bins=n_bins)
        hist.plot("Histogram for %ss" % reaction)


def plot_text_char_length_histogram(n_bins: int = N_BINS, maximum_length: int = None) -> None:
    texts = a_util.get_data_texts()
    labels = [len(t) for t in texts if maximum_length is None or len(t) < maximum_length]
    hist = a_util.Histogram(labels, n_bins=n_bins)
    if maximum_length is not None:
        title = "Histogram for Confession Text Length (in Characters) Up to %d Chars" % maximum_length
    else:
        title = "Histogram for Confession Text Length (in Characters)"
    hist.plot(title)


def plot_2d_histograms(
        n_bins: int = N_BINS,
        min_rxn: int = None,
        max_char_count: int = None,
        save: bool = False) -> None:
    # do char length and reaction  count
    all_char_lengths = [len(t) for t in a_util.get_data_texts()]
    all_rxn_labels = a_util.get_data_labels()
    label_lists = []
    for index in range(N_REACTIONS):
        char_lengths, rxn_labels = [], []
        for char_length, label in zip(all_char_lengths, all_rxn_labels):
            higher_than_min_rxn = (min_rxn is None or label[index] >= min_rxn)
            lower_than_max_char_count = (max_char_count is None or char_length <= max_char_count)
            if higher_than_min_rxn and lower_than_max_char_count:
                char_lengths.append(char_length)
                rxn_labels.append(label[index])
        label_lists.append((char_lengths, rxn_labels))
    for reaction_name, bundle in zip(REACTION_NAMES, label_lists):
        char_lengths, rxn_labels = bundle
        hist = a_util.Histogram2D(char_lengths, rxn_labels, n_bins=n_bins)
        title = "2D Hist for chr-count (X) and %ss (Y)" % reaction_name
        if min_rxn is not None:
            title = "%s %s" % (title, "with min rxn %d" % min_rxn)
        max_length_suffix = "" if (max_char_count is None) else "%d" % max_char_count
        mix_rxn_suffix = "" if (min_rxn is None) else "%d" % min_rxn
        fname = "char_count%s_v_%s%s" % (max_length_suffix, reaction_name, mix_rxn_suffix)
        hist.plot(title, save=save, fname=fname)
        print(fname)


if __name__ == '__main__':
    # plot_histograms()
    plot_2d_histograms(n_bins=100, min_rxn=5, max_char_count=400, save=True)
    # plot_2d_histograms(n_bins=100, max_char_count=800, save=True)
    # plot_text_char_length_histogram(100, maximum_length=1000)\
